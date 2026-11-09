"""Test- and train-time surveillance observation noise in physical space."""

from __future__ import annotations

import numpy as np
import torch

from merit.state import recompute_normalized_kinematics


def inject_observation_noise(
    batch_x: torch.Tensor,
    std_m: float,
    channels: list[int],
    scaler,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add zero-mean Gaussian position noise with a physical standard deviation."""
    if std_m <= 0 or scaler is None:
        return batch_x

    noisy = batch_x.clone()
    for channel in channels:
        normalized_std = float(std_m) / float(scaler.scale_[channel])
        noise = torch.randn(
            noisy.shape[0],
            noisy.shape[1],
            device=noisy.device,
            dtype=noisy.dtype,
            generator=generator,
        )
        noisy[:, :, channel] += noise * normalized_std
    mean = torch.as_tensor(
        scaler.mean_,
        device=noisy.device,
        dtype=noisy.dtype,
    )
    scale = torch.as_tensor(
        scaler.scale_,
        device=noisy.device,
        dtype=noisy.dtype,
    )
    return recompute_normalized_kinematics(noisy, mean, scale)




def inject_adsb_noise(
    batch_x: torch.Tensor,
    scale_k: float,
    scaler,
    along: torch.Tensor,
    cross: torch.Tensor,
    *,
    staleness: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    return_age: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Add measured ADS-B errors (resampled along/cross-track pairs) times scale_k.

    The local frame follows the input's own velocity channels; with clean inputs
    this is the true heading. With ``staleness`` ([M, T] seconds since the shown
    position was reported, sampled from real replay windows), each step shows the
    clean position at t - age, and a repeated report keeps its error.
    """
    if (scale_k <= 0 and staleness is None) or scaler is None:
        return (batch_x, batch_x.new_zeros(batch_x.shape[:2])) if return_age else batch_x
    mean = torch.as_tensor(scaler.mean_, device=batch_x.device, dtype=batch_x.dtype)
    scale = torch.as_tensor(scaler.scale_, device=batch_x.device, dtype=batch_x.dtype)
    b, t = batch_x.shape[:2]
    index = torch.randint(0, len(along), (b, t), device=batch_x.device, generator=generator)
    position = batch_x[:, :, :2] * scale[:2] + mean[:2]
    velocity = batch_x[:, :, 2:4] * scale[2:4] + mean[2:4]
    age = batch_x.new_zeros((b, t)) if return_age else None
    if staleness is not None:
        age = staleness[torch.randint(0, len(staleness), (b,), device=batch_x.device, generator=generator)]
        source = (torch.arange(t, device=batch_x.device, dtype=batch_x.dtype) - age).clamp(0, t - 1)
        lo = source.floor().long()
        hi = (lo + 1).clamp(max=t - 1)
        w = (source - lo)[..., None]
        gather = lambda x, i: x.gather(1, i[..., None].expand(-1, -1, 2))
        position = gather(position, lo) * (1 - w) + gather(position, hi) * w
        velocity = gather(velocity, lo)
        event = torch.cat((torch.zeros_like(source[:, :1], dtype=torch.long),
                           (source.diff(dim=1) != 0).long().cumsum(1)), dim=1)
        index = index.gather(1, event)
    speed = velocity.norm(dim=-1, keepdim=True)
    u = torch.where(speed > 1e-3, velocity / speed.clamp_min(1e-3),
                    velocity.new_tensor([1.0, 0.0]).expand_as(velocity))
    v = torch.stack((-u[..., 1], u[..., 0]), dim=-1)
    error = (along[index][..., None] * u + cross[index][..., None] * v) * float(scale_k)
    noisy = batch_x.clone()
    if staleness is None:
        noisy[:, :, :2] += error / scale[:2]
    else:
        noisy[:, :, :2] = (position + error - mean[:2]) / scale[:2]
    result = recompute_normalized_kinematics(noisy, mean, scale)
    return (result, age) if return_age else result

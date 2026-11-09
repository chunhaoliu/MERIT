"""Consistent six-channel planar trajectory-state transformations."""

from __future__ import annotations

import torch


STATE_CHANNELS = ("x", "y", "vx", "vy", "ax", "ay")


def finite_difference_state(
    position_m: torch.Tensor,
    *,
    sample_interval_seconds: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive velocity and acceleration from [B, T, 2] positions."""
    if position_m.size(1) < 2:
        zeros = torch.zeros_like(position_m)
        return zeros, zeros

    velocity = torch.empty_like(position_m)
    velocity[:, 1:, :] = (
        position_m[:, 1:, :] - position_m[:, :-1, :]
    ) / sample_interval_seconds
    velocity[:, 0, :] = velocity[:, 1, :]

    acceleration = torch.zeros_like(position_m)
    if position_m.size(1) >= 3:
        acceleration[:, 1:, :] = (
            velocity[:, 1:, :] - velocity[:, :-1, :]
        ) / sample_interval_seconds
        acceleration[:, 0, :] = acceleration[:, 1, :]
    return velocity, acceleration


def recompute_normalized_kinematics(
    state: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    *,
    sample_interval_seconds: float = 1.0,
) -> torch.Tensor:
    """Rebuild normalized vx/vy/ax/ay from normalized x/y positions."""
    if state.size(-1) < 6:
        return state
    position_m = state[:, :, :2] * scale[:2] + mean[:2]
    velocity, acceleration = finite_difference_state(
        position_m,
        sample_interval_seconds=sample_interval_seconds,
    )
    rebuilt = state.clone()
    rebuilt[:, :, 2:4] = (velocity - mean[2:4]) / scale[2:4]
    rebuilt[:, :, 4:6] = (acceleration - mean[4:6]) / scale[4:6]
    return rebuilt

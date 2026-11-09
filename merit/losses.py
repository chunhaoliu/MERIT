"""Paper forecast loss on standardized six-channel states."""
import torch


def forecast_loss(prediction, target):
    weights = torch.full((prediction.shape[-1],), 1.0, dtype=prediction.dtype, device=prediction.device)
    weights[:2] = 2.0
    return (torch.mean((prediction - target) ** 2 * weights)
            + 0.5 * (prediction[:, -1, :2] - target[:, -1, :2]).square().mean())

"""Forecast-origin-local position preprocessing.

Every method sees only the observed prefix. Window smoothers can revise earlier
observations using later observations INSIDE that prefix; they are not claimed
to be zero-lag streaming filters. No measurement uncertainty is estimated here.
"""

import torch

from merit.state import finite_difference_state


def position_to_state(position, mean, scale):
    """Build normalized six-channel state solely from physical prefix positions."""
    if position.ndim != 3 or position.shape[-1] != 2 or position.shape[1] < 2:
        raise ValueError("Expected [B,T>=2,2] physical positions")
    velocity, acceleration = finite_difference_state(position)
    return (torch.cat((position, velocity, acceleration), dim=-1) - mean) / scale


def cv_filter_history(position, process_std=5.0, measurement_std=50.0):
    """Fixed CV KF, dt=1 s, Joseph covariance update, zero-velocity prior.

    R is fixed across noise conditions (no oracle noise-level input). Initial
    velocity SD is 100 m/s. The first observed position initializes the state;
    it is not applied a second time as an independent update.
    """
    if process_std <= 0 or measurement_std <= 0:
        raise ValueError("KF standard deviations must be positive")
    b, t, _ = position.shape
    tensor = lambda value: torch.as_tensor(value, dtype=position.dtype, device=position.device)
    transition = tensor([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    observation = tensor([[1, 0, 0, 0], [0, 1, 0, 0]])
    acceleration_map = tensor([[0.5, 0], [0, 0.5], [1, 0], [0, 1]])
    q = acceleration_map @ acceleration_map.T * process_std**2
    r = torch.eye(2, device=position.device, dtype=position.dtype) * measurement_std**2
    identity = torch.eye(4, device=position.device, dtype=position.dtype)
    covariance = torch.diag(tensor([measurement_std**2] * 2 + [100.0**2] * 2))
    state = torch.cat((position[:, 0], torch.zeros_like(position[:, 0])), dim=-1)
    history = [position[:, 0]]
    # Covariances are shared because all sequences have the same sample times/R.
    for index in range(1, t):
        state = state @ transition.T
        covariance = transition @ covariance @ transition.T + q
        innovation_cov = observation @ covariance @ observation.T + r
        gain = torch.linalg.solve(innovation_cov, observation @ covariance).T
        state = state + (position[:, index] - state @ observation.T) @ gain.T
        residual = identity - gain @ observation
        covariance = residual @ covariance @ residual.T + gain @ r @ gain.T
        history.append(state[:, :2])
    return torch.stack(history, dim=1)


def preprocess(position, method, *, strength=25.0, cap=0.75):
    if method == "prefix":
        return position
    if method != "projection":
        raise ValueError(f"Unsupported preprocessing method: {method}")
    if strength < 0 or not 0 <= cap <= 1:
        raise ValueError("Invalid projection strength or cap")
    n = position.shape[1]
    identity = torch.eye(n, device=position.device, dtype=position.dtype)
    d2 = torch.diff(identity, n=2, dim=0)
    projection = torch.linalg.solve(identity + strength * d2.T @ d2, identity)
    return position + cap * (projection @ position - position)

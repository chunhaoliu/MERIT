"""Recurrent predictor and learned refinement used in the MERIT comparisons."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from merit.state import recompute_normalized_kinematics


class RecurrentCore(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = 60, 30, 6
        self.d_model, self.e_layers = configs.d_model, configs.e_layers
        self.dropout, self.blend_cap = configs.dropout, 0.75
        self.motion_projection_lambda = 25.0
        self.pos_smoother = nn.Conv1d(2, 2, 7, groups=2, bias=False)
        gate_hidden = max(self.d_model // 8, 16)
        self.reliability = nn.Sequential(
            nn.Linear(12, gate_hidden), nn.GELU(), nn.Dropout(self.dropout),
            nn.Linear(gate_hidden, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1), nn.Sigmoid())
        self.lstm = nn.LSTM(6, self.d_model, num_layers=self.e_layers, batch_first=True,
                            dropout=self.dropout if self.e_layers > 1 else 0.0)
        # Retained frozen parameters preserve the paper initialization and state dictionaries.
        self.proj = nn.Linear(self.d_model, 6)
        self.direct_proj = nn.Linear(self.d_model, 30 * 6)
        self.register_buffer("_state_mean", torch.zeros(6), persistent=False)
        self.register_buffer("_state_scale", torch.ones(6), persistent=False)
        self.register_buffer("_scaler_ready", torch.tensor(False), persistent=False)
        self.register_buffer("_motion_projection", self._build_motion_projection(60), persistent=True)
        self._init_smoother()
        self.pos_smoother.weight.requires_grad_(False)

    def _build_motion_projection(self, length: int) -> torch.Tensor:
        """Return the closed-form second-difference regularized projection."""
        identity = torch.eye(length)
        if length < 3 or self.motion_projection_lambda <= 0:
            return identity
        second_difference = torch.zeros(length - 2, length)
        index = torch.arange(length - 2)
        second_difference[index, index] = 1.0
        second_difference[index, index + 1] = -2.0
        second_difference[index, index + 2] = 1.0
        system = (
            identity
            + self.motion_projection_lambda
            * second_difference.transpose(0, 1)
            @ second_difference
        )
        return torch.linalg.solve(system, identity)

    def _init_smoother(self) -> None:
        with torch.no_grad():
            kernel_size = self.pos_smoother.kernel_size[0]
            self.pos_smoother.weight.fill_(1.0 / kernel_size)

    def _smooth_positions(self, pos):
        projection = self._motion_projection.to(device=pos.device, dtype=pos.dtype)
        return torch.einsum("ts,bsc->btc", projection, pos)

    def _local_motion_inconsistency(self, pos: torch.Tensor) -> torch.Tensor:
        """Return acceleration-like inconsistency without conflating it with speed."""
        if pos.size(1) < 3:
            return torch.zeros(pos.size(0), pos.size(1), 1, device=pos.device, dtype=pos.dtype)
        second_difference = pos[:, 2:, :] - 2.0 * pos[:, 1:-1, :] + pos[:, :-2, :]
        inconsistency = second_difference.pow(2).sum(dim=-1, keepdim=True)
        return F.pad(inconsistency, (0, 0, 2, 0), mode="replicate")

    def _local_velocity_disagreement(
        self,
        pos: torch.Tensor,
        pos_smooth: torch.Tensor,
    ) -> torch.Tensor:
        if pos.size(1) < 2:
            return torch.zeros(
                pos.size(0),
                pos.size(1),
                1,
                device=pos.device,
                dtype=pos.dtype,
            )
        raw_velocity = pos[:, 1:, :] - pos[:, :-1, :]
        smooth_velocity = pos_smooth[:, 1:, :] - pos_smooth[:, :-1, :]
        disagreement = (raw_velocity - smooth_velocity).pow(2).sum(
            dim=-1,
            keepdim=True,
        )
        return F.pad(disagreement, (0, 0, 1, 0), mode="replicate")

    def _predict_reliability(
        self,
        x_enc: torch.Tensor,
        pos: torch.Tensor,
        pos_smooth: torch.Tensor,
    ) -> torch.Tensor:
        raw_inconsistency = torch.log1p(
            self._local_motion_inconsistency(pos)
        )
        smoother_residual = torch.log1p(
            (pos_smooth - pos).pow(2).sum(dim=-1, keepdim=True)
        )
        smooth_inconsistency = torch.log1p(
            self._local_motion_inconsistency(pos_smooth)
        )
        velocity_disagreement = torch.log1p(
            self._local_velocity_disagreement(pos, pos_smooth)
        )
        gate_features = torch.cat(
            [
                raw_inconsistency,
                smoother_residual,
                smooth_inconsistency,
                velocity_disagreement,
            ],
            dim=-1,
        )
        sequence_features = torch.cat(
            [
                gate_features.mean(dim=1),
                gate_features.std(dim=1, unbiased=False),
                gate_features.amax(dim=1),
            ],
            dim=-1,
        )
        reliability = self.reliability(sequence_features).unsqueeze(1)
        return reliability.expand(-1, pos.size(1), pos.size(-1))

    def set_scaler_stats(self, mean, scale) -> None:
        mean_tensor = torch.as_tensor(
            mean,
            dtype=self._state_mean.dtype,
            device=self._state_mean.device,
        )
        scale_tensor = torch.as_tensor(
            scale,
            dtype=self._state_scale.dtype,
            device=self._state_scale.device,
        )
        self._state_mean.copy_(mean_tensor)
        self._state_scale.copy_(scale_tensor.clamp_min(1e-6))
        self._scaler_ready.fill_(True)

    def refinement_components(self, x_enc):
        pos = x_enc[:, :, :2]
        pos_smooth = self._smooth_positions(pos)
        reliability = self._predict_reliability(x_enc, pos, pos_smooth)
        blend = (1.0 - reliability) * self.blend_cap
        normalized = (blend / self.blend_cap).clamp(0.0, 1.0)
        stable = normalized.clamp(1e-5, 1.0 - 1e-5)
        normalized = torch.sigmoid(torch.logit(stable) * 2.0)
        return pos, pos_smooth, normalized * self.blend_cap

    def apply_refinement_components(
        self,
        x_enc: torch.Tensor,
        pos: torch.Tensor,
        pos_smooth: torch.Tensor,
        blend: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one already-computed refinement decision to the full state."""
        refined = x_enc.clone()
        refined[:, :, :2] = pos + blend * (pos_smooth - pos)
        if bool(self._scaler_ready):
            refined = recompute_normalized_kinematics(
                refined,
                self._state_mean,
                self._state_scale,
            )
        return refined

    def optimal_blend_target(self, pos, pos_smooth, pos_clean):
        correction = pos_smooth - pos
        numerator = ((pos_clean - pos) * correction).sum(dim=-1, keepdim=True)
        denominator = correction.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        numerator = numerator.sum(dim=1, keepdim=True)
        denominator = denominator.sum(dim=1, keepdim=True)
        return (numerator / denominator).clamp(0.0, self.blend_cap).expand_as(pos)

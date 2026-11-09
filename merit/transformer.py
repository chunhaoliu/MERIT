"""Time-step-token Transformer encoder with the same interface and endpoint residual as the LSTM rows."""
import torch
from torch import nn

from merit.inputs import position_to_state


class TrajectoryTransformer(nn.Module):
    def __init__(self, mean, scale, *, d_model=128, heads=4, layers=3, ff=256, dropout=0.1, history=60, horizon=30):
        super().__init__()
        self.register_buffer("state_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("state_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.embed = nn.Linear(6, d_model)
        self.position = nn.Parameter(torch.zeros(1, history, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model * 2, horizon * 6)
        self.horizon = horizon
        nn.init.normal_(self.position, std=0.02)

    def forward(self, observed, clean=None):
        if observed.ndim != 3 or observed.shape[1:] != (60, 6):
            raise ValueError("Expected [B,60,6] observed prefix")
        position = observed[:, :, :2] * self.state_scale[:2] + self.state_mean[:2]
        local = position_to_state(position, self.state_mean, self.state_scale)
        h = self.norm(self.encoder(self.embed(local) + self.position))
        pooled = torch.cat([h[:, -1], h.mean(1)], dim=-1)
        prediction = self.head(pooled).reshape(-1, self.horizon, 6)
        prediction = torch.cat([prediction[:, :, :2] + local[:, -1:, :2], prediction[:, :, 2:]], dim=-1)
        return prediction, observed.new_zeros(())

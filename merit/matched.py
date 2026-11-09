"""Shared initialization for the three recurrent predictors."""

from types import SimpleNamespace

import torch
from torch import nn

from merit.recurrent import RecurrentCore



ROWS = ("direct", "projection", "gate_analytic")


class MatchedPredictor(nn.Module):
    def __init__(self, row, mean, scale, *, hidden=256, layers=2, dropout=0.12):
        super().__init__()
        if row not in ROWS:
            raise ValueError(row)
        self.row = row
        self.core = RecurrentCore(SimpleNamespace(d_model=hidden, e_layers=layers, dropout=dropout))
        self.core.set_scaler_stats(mean, scale)
        for name, parameter in self.core.named_parameters():
            parameter.requires_grad_(name.startswith(("lstm.", "direct_proj.")) or
                                     (row.startswith("gate_") and name.startswith("reliability.")))

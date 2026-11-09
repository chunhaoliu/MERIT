"""Task-adapted supervised PatchTST for the MERIT comparison."""

import torch
from torch import nn

from merit.vendor_patchtst.PatchTST_backbone import PatchTST_backbone
from merit.inputs import position_to_state


ARCHITECTURE = dict(c_in=6, context_window=60, target_window=30, patch_len=16,
    stride=8, n_layers=3, d_model=128, n_heads=16, d_ff=256, norm="BatchNorm",
    attn_dropout=0., dropout=.05, act="gelu", res_attention=True, pre_norm=False,
    pe="zeros", learn_pe=True, fc_dropout=.05, head_dropout=0., padding_patch="end",
    pretrain_head=False, head_type="flatten", individual=False, revin=True,
    affine=False, subtract_last=False)


class PatchTSTReference(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("state_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("state_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.core = PatchTST_backbone(**ARCHITECTURE)

    def forward(self, observed, clean=None):
        if observed.ndim != 3 or observed.shape[1:] != (60, 6):
            raise ValueError("Expected [B,60,6] observed prefix")
        position = observed[:, :, :2]*self.state_scale[:2]+self.state_mean[:2]
        local = position_to_state(position, self.state_mean, self.state_scale)
        prediction = self.core(local.transpose(1, 2)).transpose(1, 2)
        return prediction, observed.new_zeros(())

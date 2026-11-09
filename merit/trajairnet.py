"""Single-aircraft, horizontal TrajAirNet for the frozen MERIT 60/30 interface.

Source: Patrikar et al., ICRA 2022, doi:10.1109/ICRA46639.2022.9811972.
Reference: https://github.com/castacks/trajairnet at
5a5bc751dabba09a668ca9acb6b05ae4112c5b3d, model/{trajairnet,tcn_model,
cvae_base,gat_model,gat_layers}.py. Independently expressed batched adaptation.
Upstream BSD-4-Clause notice retained below.

Copyright (c) 2021, AirLab
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software must
   display the following acknowledgement:
   This product includes software developed by [project].
4. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY COPYRIGHT HOLDER "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

No weather or neighbouring aircraft are available. Each batch item is its own
one-node graph, whose attention softmax is exactly one. The learned 16-head
projections, ELU and final tanh remain, with no unused attention-score weights.
TCNs retain [256,256,horizon], kernel 4, dilations [1,2,4], dropout .05,
weight normalization and tanh residuals. CVAE uses two 128-wide layers, 128
latent dimensions, tanh decoder and a 32-wide per-coordinate output feature.
Only positions enter the network. Targets enter the posterior only in
training_loss. forward always uses the N(0,I) prior mean z=0, never best-of-K.
The original positional RMSE plus latent-summed KL is averaged over aircraft.
Verlet uses the last TWO 1-Hz observations (paper Eq. 7), correcting upstream
acc_to_abs's first/last-history initialization for this sampling interval.
The six-channel output rebuilds physical backward differences, then uses the
unchanged training scaler. Absolute normalized coordinates are preserved.
Dilated convolutions use the equivalent unfold/linear algebra to avoid a slow
deterministic cuDNN backward path on Windows; weights and receptive fields match.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from merit.inputs import position_to_state


class TemporalBlock(nn.Module):
    def __init__(self, inputs, outputs, dilation):
        super().__init__()
        self.padding = 3 * dilation
        self.conv1 = weight_norm(nn.Conv1d(inputs, outputs, 4, padding=self.padding, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(outputs, outputs, 4, padding=self.padding, dilation=dilation))
        self.residual = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()
        self.dropout = nn.Dropout(.05)
        if isinstance(self.residual, nn.Conv1d):
            nn.init.normal_(self.residual.weight, std=.01)

    def forward(self, x):
        h = self.dropout(F.relu(self.convolve(self.conv1, x)))
        h = self.dropout(F.relu(self.convolve(self.conv2, h)))
        return torch.tanh(h + self.residual(x))

    def convolve(self, layer, x):
        if layer.dilation[0] == 1:
            return layer(x)[..., :-self.padding]
        windows = F.pad(x, (self.padding, 0)).unfold(-1, self.padding + 1, 1)[..., ::layer.dilation[0]]
        windows = windows.permute(0, 2, 1, 3).reshape(-1, x.shape[1] * 4)
        result = F.linear(windows, layer.weight.flatten(1), layer.bias)
        return result.reshape(x.shape[0], x.shape[2], -1).transpose(1, 2)


class TrajAirNet(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer("state_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("state_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.history_encoder = nn.Sequential(TemporalBlock(2, 256, 1), TemporalBlock(256, 256, 2),
                                             TemporalBlock(256, 30, 4))
        self.future_encoder = nn.Sequential(TemporalBlock(2, 256, 1), TemporalBlock(256, 256, 2),
                                            TemporalBlock(256, 30, 4))
        # Exactly the singleton-graph GAT algebra, batched without cross-window edges.
        self.self_heads = nn.Linear(60 * 30, 16 * 256, bias=False)
        self.self_output = nn.Linear(16 * 256, 60 * 30, bias=False)
        for head in self.self_heads.weight.split(256):
            nn.init.xavier_uniform_(head, gain=1.414)
        nn.init.xavier_uniform_(self.self_output.weight, gain=1.414)
        self.posterior = nn.Sequential(nn.Linear(30 * 30 + 2 * 60 * 30, 128), nn.ReLU(),
                                       nn.Linear(128, 128), nn.ReLU())
        self.posterior_mean = nn.Linear(128, 128)
        self.posterior_logvar = nn.Linear(128, 128)
        self.decoder = nn.Sequential(nn.Linear(128 + 2 * 60 * 30, 128), nn.Tanh(),
                                     nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 2 * 32), nn.Tanh())
        self.acceleration = nn.Linear(32, 30)
        nn.init.normal_(self.acceleration.weight, std=.05)

    def condition(self, observed):
        if observed.ndim != 3 or observed.shape[1:] != (60, 6):
            raise ValueError("Expected [B,60,6] observed prefix")
        encoded = self.history_encoder(observed[:, :, :2].transpose(1, 2)).flatten(1)
        social = torch.tanh(self.self_output(F.elu(self.self_heads(encoded))))
        return torch.cat((encoded, social), dim=-1)

    def decode(self, observed, condition, z):
        features = self.decoder(torch.cat((z, condition), dim=-1)).reshape(-1, 2, 32)
        acceleration = self.acceleration(features).transpose(1, 2)
        # At dt=1, two cumulative sums implement Eq. 7 without an in-place loop.
        velocity = observed[:, -1:, :2] - observed[:, -2:-1, :2]
        positions = observed[:, -1:, :2] + (velocity + acceleration.cumsum(1)).cumsum(1)
        prefix = observed[:, -2:, :2]
        physical = torch.cat((prefix, positions), dim=1) * self.state_scale[:2] + self.state_mean[:2]
        return position_to_state(physical, self.state_mean, self.state_scale)[:, 2:]

    def forward(self, observed, clean=None):
        condition = self.condition(observed)
        prediction = self.decode(observed, condition, observed.new_zeros(len(observed), 128))
        return prediction, observed.new_zeros(())

    def training_loss(self, observed, target):
        if not self.training:
            raise RuntimeError("Posterior is training-only")
        condition = self.condition(observed)
        future = self.future_encoder(target[:, :, :2].transpose(1, 2)).flatten(1)
        posterior = self.posterior(torch.cat((future, condition), dim=-1))
        mean, logvar = self.posterior_mean(posterior), self.posterior_logvar(posterior)
        z = mean + torch.randn_like(mean) * torch.exp(.5 * logvar)
        prediction = self.decode(observed, condition, z)
        # Upstream's RMSE + sum KL per aircraft, then batch mean. No KL tuning.
        reconstruction = (prediction[:, :, :2] - target[:, :, :2]).square().mean((1, 2)).clamp_min(1e-12).sqrt()
        kl = -.5 * (1 + logvar - mean.square() - logvar.exp()).sum(-1)
        return (reconstruction + kl).mean()

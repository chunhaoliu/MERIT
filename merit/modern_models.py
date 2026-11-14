"""Modern baseline adapters with the repository's six-channel local interface.

The upstream WTFTP and iTransformer implementations are adapted to accept
``[B, 60, 6]`` origin-local states and return ``[B, 30, 6]`` states.  The
position channels are residualized against the last observed local position;
kinematic channels are predicted directly.  This keeps the comparison
compatible with the registered LSTM/Transformer rows without changing their
default code paths.
"""
from types import SimpleNamespace

import torch
from torch import nn

from merit.inputs import position_to_state

class _MovingAverage(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__(); self.kernel_size=kernel_size
        self.avg=nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    def forward(self, x):
        n=(self.kernel_size-1)//2
        x=torch.cat((x[:, :1, :].repeat(1,n,1), x, x[:, -1:, :].repeat(1,n,1)), dim=1)
        return self.avg(x.permute(0,2,1)).permute(0,2,1)

class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size): super().__init__(); self.moving_avg=_MovingAverage(kernel_size)
    def forward(self,x):
        trend=self.moving_avg(x); return x-trend, trend

class DLinearCore(nn.Module):
    def __init__(self, configs):
        super().__init__(); self.task_name=configs.task_name; self.seq_len=configs.seq_len; self.pred_len=configs.pred_len
        self.channels=configs.enc_in; self.individual=getattr(configs,'individual',False); self.decomp=_SeriesDecomp(getattr(configs,'moving_avg',25))
        if self.individual:
            self.linear_seasonal=nn.ModuleList([nn.Linear(self.seq_len,self.pred_len) for _ in range(self.channels)])
            self.linear_trend=nn.ModuleList([nn.Linear(self.seq_len,self.pred_len) for _ in range(self.channels)])
        else:
            self.linear_seasonal=nn.Linear(self.seq_len,self.pred_len); self.linear_trend=nn.Linear(self.seq_len,self.pred_len)
    def forecast(self,x_enc):
        seasonal,trend=self.decomp(x_enc); seasonal=seasonal.permute(0,2,1); trend=trend.permute(0,2,1)
        if self.individual:
            so=torch.zeros(seasonal.size(0),self.channels,self.pred_len,dtype=seasonal.dtype,device=seasonal.device); to=torch.zeros_like(so)
            for c in range(self.channels): so[:,c,:]=self.linear_seasonal[c](seasonal[:,c,:]); to[:,c,:]=self.linear_trend[c](trend[:,c,:])
        else: so=self.linear_seasonal(seasonal); to=self.linear_trend(trend)
        return (so+to).permute(0,2,1)

class PatchCore(nn.Module):
    def __init__(self, configs):
        super().__init__(); self.task_name=configs.task_name; self.seq_len=configs.seq_len; self.pred_len=configs.pred_len; self.channels=configs.enc_in
        self.patch_len=min(getattr(configs,'patch_len',16),self.seq_len); self.stride=getattr(configs,'stride',max(1,self.patch_len//2)); self.num_patches=1+max(0,(self.seq_len-self.patch_len)//self.stride); self.d_model=configs.d_model
        self.patch_proj=nn.Linear(self.patch_len,self.d_model); self.position=nn.Parameter(torch.zeros(1,self.num_patches,self.d_model))
        layer=nn.TransformerEncoderLayer(d_model=self.d_model,nhead=configs.n_heads,dim_feedforward=configs.d_ff,dropout=configs.dropout,activation=getattr(configs,'activation','gelu'),batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=configs.e_layers); self.dropout=nn.Dropout(configs.dropout); self.head=nn.Linear(self.num_patches*self.d_model,self.pred_len)
    def _make_patches(self,x):
        x=x.permute(0,2,1); patches=x.unfold(dimension=-1,size=self.patch_len,step=self.stride); patches=patches[:,:,:self.num_patches,:]; return patches.reshape(-1,self.num_patches,self.patch_len)
    def forecast(self,x_enc):
        b=x_enc.size(0); p=self._make_patches(x_enc); e=self.patch_proj(p)+self.position; e=self.encoder(self.dropout(e)); out=self.head(e.reshape(e.size(0),-1)); return out.reshape(b,self.channels,self.pred_len).permute(0,2,1)


def haar_dwt1d(x):
    """One-level orthonormal Haar DWT, matching pytorch_wavelets ``haar``.

    The registered WTFTP protocol uses an even 90-step sequence, so no
    boundary extension is needed in the production path.  An odd final sample
    is symmetrically repeated to keep the helper total and differentiable.
    """
    if x.ndim < 1:
        raise ValueError("DWT expects at least one dimension")
    if x.shape[-1] % 2:
        x = torch.cat((x, x[..., -1:]), dim=-1)
    even, odd = x[..., 0::2], x[..., 1::2]
    inv_sqrt2 = x.new_tensor(2.0).rsqrt()
    return (even + odd) * inv_sqrt2, (even - odd) * inv_sqrt2


def haar_idwt1d(lo, hi, length=None):
    """Inverse of :func:`haar_dwt1d` with optional odd-length crop."""
    if lo.shape != hi.shape:
        raise ValueError("Haar low/high coefficient shapes must match")
    inv_sqrt2 = lo.new_tensor(2.0).rsqrt()
    out = torch.empty((*lo.shape[:-1], lo.shape[-1] * 2), dtype=lo.dtype, device=lo.device)
    out[..., 0::2] = (lo + hi) * inv_sqrt2
    out[..., 1::2] = (lo - hi) * inv_sqrt2
    return out if length is None else out[..., :length]


class _WaveletAttention(nn.Module):
    """The official WTFTP EnhancedBlock plus strided convolution."""

    def __init__(self, hid_dim, init_steps, skip=True, kernel=2, stride=2, padding=1):
        super().__init__()
        self.skip = skip
        self.comp = nn.Sequential(nn.Linear(hid_dim, hid_dim // 2, bias=False),
                                  nn.ReLU(), nn.Linear(hid_dim // 2, 1, bias=False))
        self.activate = nn.Sequential(nn.Linear(init_steps, init_steps // 2, bias=False),
                                      nn.ReLU(), nn.Linear(init_steps // 2, init_steps, bias=False),
                                      nn.Sigmoid())
        self.convs = nn.Sequential(nn.Conv1d(hid_dim, hid_dim, kernel, stride=stride,
                                             padding=padding, bias=False), nn.ReLU())

    def forward(self, hid_set):
        scores = self.comp(hid_set)
        weights = self.activate(scores.transpose(1, 2))
        reweighted = hid_set * weights.transpose(1, 2).expand_as(hid_set)
        if self.skip:
            reweighted = reweighted + hid_set
        aligned = self.convs(reweighted.transpose(1, 2)).transpose(1, 2)
        return aligned, weights


class WTFTPWaveletCore(nn.Module):
    """Faithful official WTFTP (attention, coefficient decoders, Haar DWT)."""

    def __init__(self, channels=6, hidden=64, enc_layers=4, dec_layers=1,
                 maxlevel=1, input_steps=89):
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.maxlevel = maxlevel
        self.input_steps = input_steps
        self.inp_embding = nn.Sequential(nn.Linear(channels, hidden // 2, bias=False), nn.ReLU(),
                                         nn.Linear(hidden // 2, hidden, bias=False), nn.ReLU())
        self.encoder = nn.LSTM(hidden, hidden, enc_layers, batch_first=True, dropout=0.0, bias=False)
        # Official defaults: maxlevel=1, Haar, kernel=2/stride=2/padding=1.
        self.Wt_attns = nn.ModuleList([_WaveletAttention(hidden, input_steps, True, 2, 2, 1)
                                       for _ in range(maxlevel + 1)])
        self.decoders = nn.ModuleList([nn.LSTM(hidden, hidden, dec_layers, batch_first=True,
                                               dropout=0.0, bias=False)
                                       for _ in range(maxlevel + 1)])
        self.LNs = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(maxlevel + 1)])
        self.oup_embdings = nn.ModuleList([nn.Linear(hidden, channels, bias=True)
                                           for _ in range(maxlevel + 1)])
        self.dec_layers = dec_layers

    def forward(self, inp):
        if inp.ndim != 3 or inp.shape[1] != self.input_steps or inp.shape[2] != self.channels:
            raise ValueError(f"Expected [B,{self.input_steps},{self.channels}] WTFTP input")
        emb = self.inp_embding(inp)
        en_oup, (_, c_en) = self.encoder(emb)
        coeffs, scores = [], []
        for attn, decoder, ln, head in zip(self.Wt_attns, self.decoders, self.LNs, self.oup_embdings):
            dec_in, weight = attn(en_oup)
            h = emb.new_zeros(self.dec_layers, inp.shape[0], self.hidden)
            c = emb.new_zeros(self.dec_layers, inp.shape[0], self.hidden)
            c[0] = c_en[-1]
            dec, _ = decoder(dec_in, (h, c))
            coeffs.append(head(ln(dec)))
            scores.append(weight)
        return coeffs, torch.cat(scores, dim=1)


class _DLinearConfig:
    task_name = "short_term_forecast"
    seq_len = 60
    pred_len = 30
    enc_in = 6
    individual = False
    moving_avg = 25


class InvertedTransformerCore(nn.Module):
    """Compact translation of the official iTransformer encoder path."""

    def __init__(self, seq_len=60, pred_len=30, d_model=128, heads=4, layers=3,
                 ff=256, dropout=.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model, bias=True)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers,
                                              norm=nn.LayerNorm(d_model))
        self.projector = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # Official iTransformer: B,L,N -> B,N,L -> B,N,E -> B,N,S -> B,S,N.
        return self.projector(self.encoder(self.value_embedding(x.transpose(1, 2)))).transpose(1, 2)


class WTFTPAdapterCore(nn.Module):
    """WTFTP's official LSTM decoder without its external data/wavelet package.

    ``maxlevel=0`` and ``steps=[30]`` produce one 30-step coefficient stream.
    This is the official ``WTFTP_AttnRemoved`` branch with the repository
    horizon substituted for the upstream recursive wavelet reconstruction.
    """

    def __init__(self, channels=6, hidden=64, enc_layers=4, dec_layers=1, horizon=30):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(channels, hidden // 2, bias=False), nn.ReLU(),
                                 nn.Linear(hidden // 2, hidden, bias=False), nn.ReLU())
        self.encoder = nn.LSTM(hidden, hidden, enc_layers, batch_first=True,
                               dropout=0.0, bias=False)
        self.decoder = nn.LSTM(hidden, hidden, dec_layers, batch_first=True,
                               dropout=0.0, bias=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, channels)
        self.horizon = horizon
        self.dec_layers = dec_layers

    def forward(self, x):
        embedded = self.inp(x)
        _, (_, c) = self.encoder(embedded)
        h = embedded.new_zeros(self.dec_layers, x.size(0), embedded.size(-1))
        c_dec = embedded.new_zeros(self.dec_layers, x.size(0), embedded.size(-1))
        c_dec[0] = c[-1]
        # Feeding the complete zero-token sequence is algebraically identical
        # to the upstream step-by-step loop, while avoiding a Python loop in
        # every training batch.
        tokens = embedded.new_zeros(x.size(0), self.horizon, embedded.size(-1))
        decoded, _ = self.decoder(tokens, (h, c_dec))
        return self.head(self.norm(decoded))


class ModernPredictor(nn.Module):
    """Common wrapper; ``row`` is dlinear/itransformer/wtftp/patchtst_r."""

    def __init__(self, row, mean, scale, config=None):
        super().__init__()
        if row not in {"dlinear", "itransformer", "wtftp", "wtftp_w", "patchtst_r"}:
            raise ValueError(row)
        config = dict(config or {})
        self.row = row
        self.register_buffer("state_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("state_scale", torch.as_tensor(scale, dtype=torch.float32))
        if row == "dlinear":
            dcfg = _DLinearConfig()
            dcfg.individual = bool(config.get("individual", False))
            dcfg.moving_avg = int(config.get("moving_avg", 25))
            self.core = DLinearCore(dcfg)
        elif row == "itransformer":
            self.core = InvertedTransformerCore(
                d_model=int(config.get("d_model", 128)),
                heads=int(config.get("heads", 4)),
                layers=int(config.get("layers", 3)),
                ff=int(config.get("ff", 256)),
                dropout=float(config.get("dropout", .1)))
        elif row == "wtftp":
            self.core = WTFTPAdapterCore(
                hidden=int(config.get("hidden", 64)),
                enc_layers=int(config.get("enc_layers", 4)),
                dec_layers=int(config.get("dec_layers", 1)))
        elif row == "wtftp_w":
            self.core = WTFTPWaveletCore(
                hidden=int(config.get("hidden", 64)),
                enc_layers=int(config.get("enc_layers", 4)),
                dec_layers=int(config.get("dec_layers", 1)),
                maxlevel=int(config.get("maxlevel", 1)), input_steps=89)
        else:
            # PatchTST-R uses the existing implementation with a local residual
            # wrapper in the training/evaluation script; this core is injected
            # by the caller to avoid importing the historical row here.
            cfg = SimpleNamespace(task_name="short_term_forecast", seq_len=60,
                                  pred_len=30, enc_in=6, patch_len=16, stride=8,
                                  d_model=128, n_heads=4, d_ff=256, e_layers=3,
                                  dropout=.1, activation="gelu")
            self.core = PatchCore(cfg)

    def forward(self, observed, clean=None):
        if observed.ndim != 3 or observed.shape[1:] != (60, 6):
            raise ValueError("Expected [B,60,6] observed prefix")
        local = position_to_state(observed[:, :, :2] * self.state_scale[:2] + self.state_mean[:2],
                                  self.state_mean, self.state_scale)
        if self.row in {"dlinear", "patchtst_r"}:
            output = self.core.forecast(local)
        elif self.row == "wtftp_w":
            # Official WTFTP predicts a full history+future sequence from all
            # but its final sample.  The 29 future placeholders are zeros,
            # hence contain no future information.
            # Input, coefficient targets and reconstruction share one frame:
            # positions relative to the last observed position, so a zero
            # placeholder means "no displacement yet".
            last = local[:, -1:, :2]
            relative = torch.cat((local[:, :, :2] - last, local[:, :, 2:]), dim=-1)
            full_input = torch.cat((relative, relative.new_zeros(relative.shape[0], 29, 6)), dim=1)
            coeffs, _ = self.core(full_input)
            rec = haar_idwt1d(coeffs[-1].transpose(1, 2), coeffs[0].transpose(1, 2), 90)
            output = rec.transpose(1, 2)[:, 60:]
            aux = observed.new_zeros(())
            if clean is not None and clean.shape[1:] == (90, 6):
                target = torch.cat((clean[:, :, :2] - last, clean[:, :, 2:]), dim=-1).transpose(1, 2)
                lo, hi = haar_dwt1d(target)
                aux = torch.nn.functional.mse_loss(coeffs[-1].transpose(1, 2), lo)
                aux = aux + torch.nn.functional.mse_loss(coeffs[0].transpose(1, 2), hi)
            output = output[:, :30]
            output = torch.cat((output[:, :, :2] + last, output[:, :, 2:]), dim=-1)
            return output, aux
        else:
            output = self.core(local)
        if isinstance(output, tuple):
            output = output[0]
        output = output[:, :30, :]
        output = torch.cat((output[:, :, :2] + local[:, -1:, :2], output[:, :, 2:]), dim=-1)
        return output, observed.new_zeros(())


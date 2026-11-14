"""Load released predictors and evaluate immutable raw-replay windows."""
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from merit.inputs import position_to_state
from merit.lstm import EndpointResidual
from merit.patchtst import PatchTSTReference
from merit.trajairnet import TrajAirNet
from merit.transformer import TrajectoryTransformer
from merit.modern_models import ModernPredictor


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("direct", "projection", "gate_analytic", "patchtst", "transformer", "trajairnet", "seq2seq", "wtftp", "itransformer", "dlinear", "patchtst_r")
SEEDS = (42, 123, 456, 0, 1)


def make_model(row, mean, scale, *, filter_cv=None, report_age=False, config=None):
    if row in {"seq2seq", "wtftp", "itransformer", "dlinear", "patchtst_r"}:
        core_row = "wtftp" if row == "seq2seq" else "wtftp_w" if row == "wtftp" else row
        if config is None:
            config = ({"hidden": 96, "enc_layers": 3, "dec_layers": 1} if core_row == "wtftp" else
                      {"hidden": 96, "enc_layers": 3, "dec_layers": 1, "maxlevel": 1} if core_row == "wtftp_w" else
                      {"d_model": 128, "heads": 4, "layers": 3, "ff": 256, "dropout": .1} if core_row == "itransformer" else
                      {"individual": True, "moving_avg": 25} if core_row == "dlinear" else
                      {"d_model": 128})
        return ModernPredictor(core_row, mean, scale, config=config)
    if row == "patchtst":
        return PatchTSTReference(mean, scale)
    if row == "transformer":
        return TrajectoryTransformer(mean, scale)
    if row == "trajairnet":
        return TrajAirNet(mean, scale)
    return EndpointResidual(row, mean, scale, True, filter_cv=filter_cv, report_age=report_age)


def origin_local_input(x, scaler):
    mean = torch.as_tensor(scaler.mean_, dtype=x.dtype, device=x.device)
    scale = torch.as_tensor(scaler.scale_, dtype=x.dtype, device=x.device)
    local = position_to_state(x[:, :, :2] * scale[:2] + mean[:2], mean, scale)
    local[:, :, :2] = x[:, :, :2]
    return local


def predict(checkpoint, row, history, mean, scale, *, filter_cv=None, report_age=None, config=None, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    class Scaler:
        mean_, scale_ = mean, scale

    model = make_model(row, mean, scale, filter_cv=filter_cv, report_age=report_age is not None, config=config).to(device)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
    model.eval()
    x = torch.zeros(len(history), 60, 6, device=device)
    x[:, :, :2] = torch.as_tensor((history - mean[:2]) / scale[:2], dtype=torch.float32, device=device)
    x = origin_local_input(x, Scaler)
    if report_age is not None:
        if report_age.shape != history.shape[:2] or not np.isfinite(report_age).all() or np.any(report_age < 0):
            raise ValueError("Expected finite nonnegative report ages [N,60] in seconds")
        x = torch.cat((x, torch.as_tensor(report_age, dtype=x.dtype, device=device)[..., None]), dim=-1)
    precision = (torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False)
                 if row == "trajairnet" else nullcontext())
    out = []
    with torch.inference_mode(), precision:
        for offset in range(0, len(x), 512):
            out.append(model(x[offset:offset + 512])[0][:, :, :2].double().cpu().numpy() * scale[:2] + mean[:2])
    return np.concatenate(out)

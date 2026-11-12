"""Paper observation replay, validation loss and training-window loading."""
import datetime as dt
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from merit.data import AircraftDataset
from merit.losses import forecast_loss
from merit.replay import inject_adsb_noise, inject_observation_noise


ROOT = Path(__file__).resolve().parents[1]
NOISE = (0, 25, 50, 100)


def bernoulli_ages(p, n=20000, t=60, seed=0):
    if not 0 <= p <= 1:
        raise ValueError("Repeat probability must lie in [0,1]")
    repeat = np.random.RandomState(seed).rand(n, t) < p
    repeat[:, 0] = False
    ages = np.zeros((n, t))
    for k in range(1, t):
        ages[:, k] = np.where(repeat[:, k], ages[:, k - 1] + 1, 0)
    return ages


class ObservationReplay:
    def __init__(self, mode, *, kmax=2.0, pool="measured", ages=None, report_age=False):
        self.mode, self.kmax, self.report_age = mode, kmax, report_age
        with np.load(ROOT / "merit/residual_pool.npz") as z:
            self.pool = {k: z[k] for k in ("along", "cross")}
        if pool.startswith("gaussian:"):
            rng = np.random.RandomState(0)
            std = float(pool.split(":")[1])
            if std < 0:
                raise ValueError("Gaussian standard deviation must be nonnegative")
            self.pool = {k: rng.normal(0, std, len(self.pool["along"])) for k in ("along", "cross")}
        elif pool != "measured":
            raise ValueError("Expected measured or gaussian:STD residual pool")
        self.ages = None
        if mode == "merit":
            ages = str(ages or ROOT / "merit/ages_kbtp.npz")
            if ages.startswith("bernoulli:"):
                self.ages = bernoulli_ages(float(ages.split(":")[1]))
            else:
                with np.load(ages) as z:
                    self.ages = z["changed_position_age_s"]
        self.bank = {}

    def __call__(self, x, scaler, sigma, seed):
        generator = torch.Generator(device=x.device).manual_seed(seed)
        if self.mode == "iso":
            return inject_observation_noise(x, sigma, [0, 1], scaler, generator=generator)
        if self.mode == "none" or sigma <= 0:
            return torch.cat((x, x.new_zeros((*x.shape[:2], 1))), dim=-1) if self.report_age else x
        key = (x.device, x.dtype)
        if key not in self.bank:
            self.bank[key] = tuple(torch.as_tensor(self.pool[k], device=x.device, dtype=x.dtype) for k in ("along", "cross"))
            self.bank[key, "ages"] = (None if self.ages is None
                                       else torch.as_tensor(self.ages, device=x.device, dtype=x.dtype))
        result = inject_adsb_noise(x, sigma / 100.0 * self.kmax, scaler, *self.bank[key],
                                   staleness=self.bank[key, "ages"], generator=generator, return_age=self.report_age)
        if self.report_age:
            observed, age = result
            return torch.cat((observed, age[..., None]), dim=-1)
        return result


def load_data(data, device, *, raw=False, smoke=False, date_disjoint=False):
    data = Path(data)
    cache = data / "trajair/trajair_trajectories_v3.pkl"
    if not cache.exists():
        cache = data / "data/trajair/trajair_trajectories_v3.pkl"
    manifest = json.loads((ROOT / "merit/data_manifest.json").read_text())
    if hashlib.sha256(cache.read_bytes()).hexdigest() != manifest["processed_cache_sha256"]:
        raise ValueError("Processed cache differs from the paper build")
    with cache.open("rb") as f:
        payload = pickle.load(f)
    args = SimpleNamespace(window_stride=30, max_train_windows=30000, max_val_windows=6000)
    splits, expected = payload["splits"], manifest["processed"]
    if date_disjoint:  # scenes dated inside the measurement period; the scaler is refitted on them
        splits = expected = json.loads((ROOT / "merit/date_disjoint.json").read_text())
    train = AircraftDataset(payload["trajectories"], splits, args, "train", [60, 30, 30])
    val = AircraftDataset(payload["trajectories"], splits, args, "val", [60, 30, 30])
    val.set_scaler(train.scaler)
    arrays = (train.windows, val.windows)
    for name, array in zip(("train", "val"), arrays):
        if hashlib.sha256(array.tobytes()).hexdigest() != expected[f"{name}_windows_sha256"]:
            raise ValueError(f"{name} windows differ from the paper build")
    if raw:
        replay = data / "raw_replay/train.npz"
        if hashlib.sha256(replay.read_bytes()).hexdigest() != manifest["raw_replay"]["train.npz"]["sha256"]:
            raise ValueError("Raw training windows differ from the paper build")
        with np.load(replay) as z:
            position = np.concatenate([z["history_xy_m"], z["reference_xy_m"]], axis=1)
            velocity = np.gradient(position, axis=1)
            state = np.concatenate([position, velocity, np.gradient(velocity, axis=1)], axis=-1)
            state = ((state - train.scaler.mean_) / train.scaler.scale_).astype(np.float32)
            days = sorted(set(z["day"]), key=lambda d: dt.datetime.strptime(d, "%m-%d-%y"))
            held_out = np.isin(z["day"], days[-8:])
        arrays = (state[~held_out], state[held_out])
    if smoke:
        arrays = (arrays[0][:64], arrays[1][:32])
    return tuple(torch.as_tensor(a, device=device) for a in arrays), train.scaler


def validate(model, windows, scaler, noise, batch_size=256):
    model.eval()
    sums = {n: np.zeros(3, dtype=np.float64) for n in NOISE}
    scale = torch.as_tensor(scaler.scale_[:2], device=windows.device)
    with torch.inference_mode():
        for offset in range(0, len(windows), batch_size):
            batch = windows[offset:offset + batch_size]
            x, target = batch[:, :60], batch[:, 60:]
            stacked = torch.cat([noise(x, scaler, n, 7000000 + offset * 1000 + n * 10) for n in NOISE])
            output, _ = model(stacked)
            for n, prediction in zip(NOISE, output.split(len(batch))):
                distances = ((prediction[:, :, :2].double() - target[:, :, :2].double()) * scale).norm(dim=-1)
                sums[n] += [float(forecast_loss(prediction, target)) * len(batch),
                            float(distances.sum()), float(distances[:, -1].sum())]
    metrics = {str(n): {"loss": s[0] / len(windows), "ADE": s[1] / (len(windows) * 30),
                       "FDE": s[2] / len(windows)} for n, s in sums.items()}
    return float(np.mean([v["loss"] for v in metrics.values()])), metrics

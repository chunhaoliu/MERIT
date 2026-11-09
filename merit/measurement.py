"""Measure TrajAir raw ADS-B position-error structure on training-period days.

Residual = position minus a centred local quadratic fit over +-half seconds, on the
first appearance of each new position (airborne, gaps <= 10 s). Residuals are split
into along-/cross-track using the fitted velocity. The estimator is calibrated on
synthetic Aircraft trajectories with known AR(1) noise, because a local fit shrinks
correlated noise and distorts autocorrelation; compare fingerprints, not raw values.
Writes summary.json and the along/cross residual pool used by --train-noise adsb.
"""
import argparse
import datetime as dt
import glob
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RAW = None
SYNTHETIC = None
TRAIN_REPLAY = None
R_EARTH = 6371008.8
GAP_S, MIN_UPDATES, MIN_ALT_FT, MIN_SPEED = 10.0, 40, 1500, 20.0
LAGS = (1, 2, 3, 5, 10)


def train_days(raw):
    """First 77/111 of the chronological source days (TrajAir: 77), returned in directory-name order."""
    days = sorted(glob.glob(str(raw / "*_adsb")), key=lambda d: dt.datetime.strptime(os.path.basename(d)[:8], "%m-%d-%y"))
    return sorted(days[:round(len(days) * 77 / 111)])


def load_segments(days, gap=GAP_S, min_updates=MIN_UPDATES):
    segments, intervals = [], []
    for day in days:
        for path in glob.glob(os.path.join(day, "*.csv")):
            try:
                frame = pd.read_csv(path, usecols=["ID", "Time", "Altitude", "Lat", "Lon"])
            except pd.errors.EmptyDataError:
                continue
            stamps = pd.to_datetime(frame["Time"], format="%H:%M:%S.%f", errors="coerce")
            frame["t"] = (stamps - stamps.dt.normalize()).dt.total_seconds()
            for _, g in frame.dropna().groupby("ID", sort=False):
                g = g.sort_values("t", kind="stable")
                g = g[((g.Lat.diff() != 0) | (g.Lon.diff() != 0)) & (g.Altitude > MIN_ALT_FT)]
                if len(g) < min_updates:
                    continue
                t = g.t.to_numpy()
                lat0, lon0 = np.radians(g.Lat.mean()), np.radians(g.Lon.mean())
                x = R_EARTH * (np.radians(g.Lon.to_numpy()) - lon0) * np.cos(lat0)
                y = R_EARTH * (np.radians(g.Lat.to_numpy()) - lat0)
                for idx in np.split(np.arange(len(t)), np.where(np.diff(t) > gap)[0] + 1):
                    if len(idx) >= min_updates and np.all(np.diff(t[idx]) > 0):
                        segments.append((t[idx] - t[idx][0], x[idx], y[idx]))
                        intervals.append(np.diff(t[idx]))
    return segments, np.concatenate(intervals)


def fit_center(t, x, half):
    residual, velocity = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for i in range(len(t)):
        m = np.abs(t - t[i]) <= half
        if m.sum() < 7 or t[m].min() > t[i] - .6 * half or t[m].max() < t[i] + .6 * half:
            continue
        p = np.polyfit(t[m] - t[i], x[m], 2)
        residual[i], velocity[i] = x[i] - p[2], p[1]
    return residual, velocity


def lag_pairs(t, r, lag, clip):
    j = np.searchsorted(t, t + lag - .5)
    ok = j < len(t)
    j, i = j[ok], np.flatnonzero(ok)
    keep = (np.abs(t[j] - t[i] - lag) < .5) & np.isfinite(r[i]) & np.isfinite(r[j]) & (np.abs(r[i]) < clip) & (np.abs(r[j]) < clip)
    return r[i[keep]], r[j[keep]]


def decompose(segments, half, clip):
    out = {"along": [], "cross": []}
    pairs = {k: {lag: ([], []) for lag in LAGS} for k in out}
    for t, x, y in segments:
        rx, vx = fit_center(t, x, half)
        ry, vy = fit_center(t, y, half)
        speed = np.hypot(vx, vy)
        ok = np.isfinite(speed) & (speed > MIN_SPEED)
        ux = np.where(ok, vx / np.where(ok, speed, 1), np.nan)
        uy = np.where(ok, vy / np.where(ok, speed, 1), np.nan)
        comp = {"along": rx * ux + ry * uy, "cross": -rx * uy + ry * ux}
        for k, r in comp.items():
            out[k].append(r[np.isfinite(r)])
            for lag in LAGS:
                a, b = lag_pairs(t, r, lag, clip)
                pairs[k][lag][0].append(a)
                pairs[k][lag][1].append(b)
    return {k: np.concatenate(v) for k, v in out.items()}, pairs


def stats(r, pairs):
    corr = {}
    for lag, (a, b) in pairs.items():
        a, b = np.concatenate(a), np.concatenate(b)
        corr[str(lag)] = float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 else None
    return dict(n=int(len(r)), mad=float(1.4826 * np.median(np.abs(r - np.median(r)))),
                p95=float(np.percentile(np.abs(r), 95)), p99=float(np.percentile(np.abs(r), 99)),
                frac_over_50m=float(np.mean(np.abs(r) > 50)), acf_clipped=corr)


def synthetic(sigma, rho, n=300, seed=0):
    with Path(SYNTHETIC).open("rb") as stream:
        tracks = pickle.load(stream)["trajectories"]
    rng = np.random.RandomState(seed)
    segments = []
    for k in rng.choice(len(tracks), n, replace=False):
        t = np.arange(tracks.shape[1], dtype=float)
        noisy = []
        for c in (0, 1):
            e = np.zeros(len(t))
            e[0] = rng.normal(0, sigma)
            for i in range(1, len(t)):
                e[i] = rho * e[i - 1] + rng.normal(0, sigma * np.sqrt(1 - rho ** 2))
            noisy.append(tracks[k, :, c].astype(float) + e)
        segments.append((t, noisy[0], noisy[1]))
    return segments


def recovery(pool, halves, n=300, seed=0):
    """Estimator check: inject known along/cross errors (resampled from a pool) on synthetic tracks, re-estimate."""
    with Path(SYNTHETIC).open("rb") as stream:
        tracks = pickle.load(stream)["trajectories"]
    rng = np.random.RandomState(seed)
    segments, injected = [], {"along": [], "cross": []}
    for k in rng.choice(len(tracks), n, replace=False):
        x, y = tracks[k, :, 0].astype(float), tracks[k, :, 1].astype(float)
        vx, vy = np.gradient(x), np.gradient(y)
        speed = np.maximum(np.hypot(vx, vy), 1e-6)
        j = rng.randint(0, len(pool["along"]), len(x))
        ea, ec = pool["along"][j], pool["cross"][j]
        injected["along"].append(ea); injected["cross"].append(ec)
        segments.append((np.arange(len(x), dtype=float), x + (ea * vx - ec * vy) / speed, y + (ea * vy + ec * vx) / speed))
    true = {k: np.concatenate(v) for k, v in injected.items()}
    out = {"true": {k: dict(mad=float(1.4826 * np.median(np.abs(v - np.median(v)))), p99=float(np.percentile(np.abs(v), 99)))
                    for k, v in true.items()}}
    for half in halves:
        r, _ = decompose(segments, half, 50.0)
        out[f"half{half:g}"] = {k: dict(mad=float(1.4826 * np.median(np.abs(v - np.median(v)))),
                                        p99=float(np.percentile(np.abs(v), 99))) for k, v in r.items()}
    return out


def day_ci(days, half, clip, draws=1000, seed=0):
    """Day-cluster bootstrap 95% CIs of the Table-I statistics (residuals at one half-width + replay staleness)."""
    per_day = []
    for day in days:
        try:
            segments, _ = load_segments([day])
        except ValueError:  # no qualifying segment on this day
            continue
        r, _ = decompose(segments, half, clip)
        per_day.append(r)
    replay = np.load(TRAIN_REPLAY)
    history, age, rday = replay["history_xy_m"], replay["changed_position_age_s"], replay["day"]
    repeat = np.mean(np.all(history[:, 1:] == history[:, :-1], axis=2), axis=1)  # seconds repeating the previous position
    stale = {d: (repeat[rday == d], age[rday == d].ravel()) for d in np.unique(rday)}

    def table(rs, st):
        out = {}
        for k in ("along", "cross"):
            v = np.concatenate([r[k] for r in rs])
            out[k] = dict(mad=1.4826 * np.median(np.abs(v - np.median(v))), p95=np.percentile(np.abs(v), 95),
                          p99=np.percentile(np.abs(v), 99), frac_over_50m=np.mean(np.abs(v) > 50))
        out["replay"] = dict(repeat_fraction=np.mean(np.concatenate([s[0] for s in st])),
                             median_age_s=np.median(np.concatenate([s[1] for s in st])))
        return out

    keys = list(stale)
    point = table(per_day, [stale[k] for k in keys])
    rng = np.random.RandomState(seed)
    boots = [table([per_day[i] for i in rng.randint(0, len(per_day), len(per_day))],
                   [stale[keys[i]] for i in rng.randint(0, len(keys), len(keys))]) for _ in range(draws)]
    result = {g: {s: dict(value=float(point[g][s]), ci95=[float(np.percentile([b[g][s] for b in boots], q)) for q in (2.5, 97.5)])
                  for s in point[g]} for g in point}
    result.update(days=len(per_day), replay_days=len(keys), half=half, draws=draws)
    return result

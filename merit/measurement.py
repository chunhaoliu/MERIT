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


def _influence_segments(days, gap=GAP_S, min_updates=MIN_UPDATES, half=10.0):
    """Decompose residuals with receiver range, fitted speed and day."""
    from .coordinates import RECEIVER
    out = {k: [] for k in ("along", "cross", "range_m", "speed_mps", "altitude_m", "day")}
    lat_r, lon_r = np.radians(RECEIVER)
    for day in days:
        day_name = os.path.basename(day)[:8]
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
                xf = R_EARTH * (np.radians(g.Lon.to_numpy()) - lon_r) * np.cos(lat_r)
                yf = R_EARTH * (np.radians(g.Lat.to_numpy()) - lat_r)
                alt = g.Altitude.to_numpy(dtype=float) * 0.3048
                for idx in np.split(np.arange(len(t)), np.where(np.diff(t) > gap)[0] + 1):
                    if len(idx) < min_updates or not np.all(np.diff(t[idx]) > 0):
                        continue
                    ts = t[idx] - t[idx][0]
                    rx, vx = fit_center(ts, x[idx], half)
                    ry, vy = fit_center(ts, y[idx], half)
                    speed = np.hypot(vx, vy)
                    valid = np.isfinite(speed) & (speed > MIN_SPEED) & np.isfinite(rx) & np.isfinite(ry)
                    ux = np.divide(vx, speed, out=np.zeros_like(vx), where=speed > 0)
                    uy = np.divide(vy, speed, out=np.zeros_like(vy), where=speed > 0)
                    along, cross = rx * ux + ry * uy, -rx * uy + ry * ux
                    if not np.any(valid):
                        continue
                    out["along"].append(along[valid]); out["cross"].append(cross[valid])
                    range_m = np.hypot(xf[idx], yf[idx])
                    out["range_m"].append(range_m[valid]); out["speed_mps"].append(speed[valid])
                    out["altitude_m"].append(alt[idx][valid]); out["day"].append(np.full(valid.sum(), day_name, dtype="U8"))
    return {k: np.concatenate(v) if v else np.empty(0, dtype=("U8" if k == "day" else float)) for k, v in out.items()}


def _merge_influence_edges(values, edges, minimum=2000):
    edges = list(edges)
    while len(edges) > 2:
        counts, _ = np.histogram(values, bins=edges)
        small = np.flatnonzero(counts < minimum)
        if not len(small):
            break
        i = int(small[0])
        if i == 0:
            del edges[1]
        elif i == len(counts) - 1:
            del edges[-2]
        elif counts[i - 1] <= counts[i + 1]:
            del edges[i]
        else:
            del edges[i + 1]
    return edges


def _day_cluster(values, day, fn, draws=1000, seed=0):
    """Point value and day-cluster bootstrap 95% interval."""
    groups = [values[day == d] for d in np.unique(day)]
    if not groups:
        return {"value": None, "ci95": [None, None]}
    rng = np.random.RandomState(seed)
    boot = [fn(np.concatenate([groups[i] for i in rng.randint(0, len(groups), len(groups))])) for _ in range(draws)]
    return {"value": float(fn(values)), "ci95": [float(np.percentile(boot, q)) for q in (2.5, 97.5)]}


def influence(days, half=10.0, draws=1000, seed=0):
    """Residual spread versus receiver range and fitted speed, plus replay timing by range."""
    from .coordinates import RECEIVER, runway_geometry
    mad = lambda z: 1.4826 * np.median(np.abs(z - np.median(z)))
    res = _influence_segments(days, half=half)

    def rows(key, values, edges, day, stats):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (values >= lo) & (values < hi)
            row = {"lo": float(lo), "hi": float(hi) if np.isfinite(hi) else None, "n": int(m.sum()),
                   "median": float(np.median(values[m]))}
            for name, (v, fn) in stats.items():
                row[name] = _day_cluster(v[m], day[m], fn, draws, seed)
            out.append(row)
        return {"variable": key, "bins": out}

    resid_stats = {f"{c}_{s}": (res[c], fn) for c in ("along", "cross")
                   for s, fn in (("mad", mad), ("p95", lambda z: np.percentile(np.abs(z), 95)))}
    range_edges = _merge_influence_edges(res["range_m"], [0, 2000, 4000, 6000, 8000, 12000, np.inf])
    speed_edges = np.quantile(res["speed_mps"], np.linspace(0, 1, 6)); speed_edges[-1] = np.inf
    origin, axes = runway_geometry()
    z = np.load(TRAIN_REPLAY)
    xy, age = z["history_xy_m"], z["changed_position_age_s"]
    rng_m = np.linalg.norm(xy - (-origin) @ axes.T, axis=-1)
    rday = np.repeat(z["day"], xy.shape[1] - 1)
    rep = np.all(xy[:, 1:] == xy[:, :-1], axis=2).ravel().astype(float)
    r1, a1 = rng_m[:, 1:].ravel(), age[:, 1:].ravel()
    timing_edges = _merge_influence_edges(r1, [0, 1000, 2000, 3000, 4000, 5000, np.inf])
    timing_stats = {"median_age_s": (a1, np.median), "p90_age_s": (a1, lambda v: np.percentile(v, 90)),
                    "repeat_fraction": (rep, np.mean)}
    return {"schema": "merit_adsb_influence_v2", "receiver_lat_lon": list(RECEIVER), "range": "horizontal distance from the receiver (m)",
            "days": [os.path.basename(d)[:8] for d in days], "half_window_s": half, "draws": draws, "seed": seed,
            "residual_range": rows("range_m", res["range_m"], range_edges, res["day"], resid_stats),
            "residual_speed": rows("speed_mps", res["speed_mps"], speed_edges, res["day"], resid_stats),
            "timing_range": rows("range_m", r1, timing_edges, rday, timing_stats),
            "pooled": {"n": int(len(res["along"])), "along_mad": float(mad(res["along"])), "cross_mad": float(mad(res["cross"])),
                       "median_age_s": float(np.median(a1)), "repeat_fraction": float(rep.mean()),
                       "median_altitude_m": float(np.median(res["altitude_m"]))}}

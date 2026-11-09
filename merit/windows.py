"""Chronological raw-replay construction with processed-training overlap labels."""
import datetime as dt
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from merit import coordinates
from merit.raw import build_frame


DAYS = {"train": (0, 77), "val": (77, 88), "test": (88, 111)}
RECIPE = {"window": {"history": 60, "future": 30, "dt_s": 1, "stride_s": 30,
                     "max_input_report_age_s": 30, "max_label_bracket_gap_s": 5, "max_segment_gap_s": 30}}
SITE_RECIPE = {"kbtp": RECIPE, "kagc": {"window": dict(RECIPE["window"], max_label_bracket_gap_s=15, stride_s=10)}}


def processed_train_overlap(aircraft, history, reference, processed):
    points = {}
    for path in glob.glob(str(Path(processed) / "train/*.txt")):
        scene = np.atleast_2d(np.loadtxt(path))
        if scene.size == 0:
            continue
        for a in np.unique(scene[:, 1].astype(np.int64)):
            points.setdefault(str(a), []).append(scene[scene[:, 1].astype(np.int64) == a][:, 2:4] * 1000)
    trees = {k: cKDTree(np.concatenate(v)) for k, v in points.items()}
    return np.array([np.mean(trees[str(a)].query(np.concatenate([h, r]))[0] < 5) if str(a) in trees else 0.
                     for a, h, r in zip(aircraft, history, reference)])


def build_windows(raw, processed, split, site="kbtp"):
    days = sorted(glob.glob(str(Path(raw) / "*_adsb")),
                  key=lambda d: dt.datetime.strptime(os.path.basename(d)[:8], "%m-%d-%y"))
    expected = 111 if site == "kbtp" else 185
    if len(days) != expected:
        raise ValueError(f"Expected {expected} source days at {site}, found {len(days)}")
    coordinates.use_site(site)
    lo, hi = (round(len(days) * b / 111) for b in DAYS[split])
    parts = []
    for day in days[lo:hi]:
        for path in sorted(glob.glob(os.path.join(day, "*.csv"))):
            try:
                frame = pd.read_csv(path, usecols=["ID", "Date", "Time", "Range", "Bearing", "Altitude"], dtype={"ID": "string"})
            except pd.errors.EmptyDataError:
                continue
            arrays, counts = build_frame(frame, SITE_RECIPE[site])
            if counts["accepted_windows"]:
                arrays["day"] = np.full(counts["accepted_windows"], os.path.basename(day)[:8])
                parts.append(arrays)
    if not parts:
        raise ValueError("No eligible replay windows")
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    data["train_overlap_fraction"] = (processed_train_overlap(data["aircraft_id"], data["history_xy_m"], data["reference_xy_m"], processed)
                                      if site == "kbtp" else np.zeros(len(data["day"])))
    return data

"""Build public-source data and check SHA-256 against the paper build."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from merit.data import AircraftDataset, load_trajair_data
from merit.kagc import convert
from merit.windows import build_windows

MANIFEST = json.loads((ROOT / "merit/data_manifest.json").read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_processed(cache):
    ok = sha(cache) == MANIFEST["processed_cache_sha256"]
    print(f"processed cache bytes {'identical' if ok else 'DIFFER'}", flush=True)
    with cache.open("rb") as f:
        payload = pickle.load(f)
    args = SimpleNamespace(window_stride=30, max_train_windows=30000, max_val_windows=6000)
    train = AircraftDataset(payload["trajectories"], payload["splits"], args, "train", [60, 30, 30])
    val = AircraftDataset(payload["trajectories"], payload["splits"], args, "val", [60, 30, 30])
    val.set_scaler(train.scaler)
    for name, array in (("train", train.windows), ("val", val.windows)):
        match = hashlib.sha256(array.tobytes()).hexdigest() == MANIFEST["processed"][f"{name}_windows_sha256"]
        print(f"{name} windows ({len(array)}) {'identical' if match else 'DIFFER'}", flush=True)
        ok &= match
    ok &= train.scaler.mean_.tolist() == MANIFEST["processed"]["mean"]
    ok &= train.scaler.scale_.tolist() == MANIFEST["processed"]["scale"]
    return ok


def verify_replay(path):
    expected = MANIFEST["raw_replay"][path.name]
    ok = sha(path) == expected["sha256"]
    with np.load(path) as z:
        ok &= set(z.files) == set(expected["arrays"])
        for name, spec in expected["arrays"].items():
            array = z[name]
            ok &= (list(array.shape) == spec["shape"] and str(array.dtype) == spec["dtype"]
                   and hashlib.sha256(array.tobytes()).hexdigest() == spec["sha256"])
    print(f"{path.name} container and arrays {'identical' if ok else 'DIFFER'}", flush=True)
    return bool(ok)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajair", type=Path, help="111_days directory containing processed_data/ and raw_data/")
    p.add_argument("--kagc-extract", type=Path, help="extracted TartanAviation date directories")
    p.add_argument("--kagc-raw", type=Path, help="already converted KAGC raw_data directory")
    p.add_argument("--output", type=Path, default=Path("data"))
    p.add_argument("--verify-only", type=Path, help="read-only check of an existing prepared directory")
    p.add_argument("--sites", nargs="+", choices=["kbtp", "kagc"], default=["kbtp", "kagc"])
    p.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    a = p.parse_args()
    if a.verify_only:
        root = a.verify_only
        cache = root / "trajair/trajair_trajectories_v3.pkl"
        if not cache.exists():
            cache = root / "data/trajair/trajair_trajectories_v3.pkl"
        ok = verify_processed(cache)
        for site in a.sites:
            for split in a.splits:
                filename = f"{split}.npz" if site == "kbtp" else f"kagc_{split}.npz"
                ok &= verify_replay(root / "raw_replay" / filename)
        return 0 if ok else 1
    if a.trajair is None:
        p.error("--trajair is required when rebuilding")
    sources = [s.resolve() for s in (a.trajair, a.kagc_extract, a.kagc_raw) if s is not None]
    if any(a.output.resolve().is_relative_to(s) for s in sources):
        raise ValueError("Choose an output outside the source directories")
    if a.output.exists():
        raise ValueError("Build output already exists. Use a fresh output or --verify-only.")
    cache = a.output / "trajair/trajair_trajectories_v3.pkl"
    load_trajair_data(a.trajair, cache_path=cache)
    ok = verify_processed(cache)
    kagc_raw = a.kagc_raw
    if a.kagc_extract:
        kagc_raw = a.output / "kagc/raw_data"
        for source in sorted(a.kagc_extract.glob("*/1.csv")):
            dest = kagc_raw / f"{source.parent.name}_adsb" / "1.csv"
            convert(source, dest)
    for site in a.sites:
        raw = a.trajair / "raw_data" if site == "kbtp" else kagc_raw
        if raw is None:
            p.error("KAGC requires --kagc-raw or --kagc-extract")
        for split in a.splits:
            data = build_windows(raw, a.trajair / "processed_data", split, site)
            filename = f"{split}.npz" if site == "kbtp" else f"kagc_{split}.npz"
            target = a.output / "raw_replay" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **data)
            ok &= verify_replay(target)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

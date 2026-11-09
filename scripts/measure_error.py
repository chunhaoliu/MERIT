"""Measure residual structure, calibrate its estimator and regenerate the measured pool."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from merit import measurement as m
from merit.synthetic import generate_aircraft_trajectories


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, help="site raw_data directory")
    p.add_argument("--train-replay", type=Path, help="training-period replay NPZ for day CIs")
    p.add_argument("--synthetic-cache", type=Path, help="optional known-track cache for estimator calibration")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["measure", "recovery", "day-ci"], default="measure")
    p.add_argument("--pool", type=Path, default=ROOT / "merit/residual_pool.npz")
    p.add_argument("--halves", nargs="+", type=float, default=[5, 10, 15])
    p.add_argument("--clip", type=float, default=50.)
    p.add_argument("--gap", type=float, default=10.)
    p.add_argument("--min-updates", type=int, default=40)
    p.add_argument("--pool-half", type=float, default=10.)
    p.add_argument("--pool-days", type=int, default=20)
    p.add_argument("--max-days", type=int, default=0)
    p.add_argument("--skip-calibration", action="store_true")
    a = p.parse_args()
    if a.mode != "recovery" and a.raw is None:
        p.error("--raw is required for measurement and day CIs")
    if a.mode == "day-ci" and a.train_replay is None:
        p.error("--train-replay is required for day CIs")
    for source in (a.raw, a.train_replay, a.synthetic_cache):
        if source is not None and (a.output.resolve() == source.resolve() or a.output.resolve().is_relative_to(source.resolve())):
            raise ValueError("Choose a measurement output outside the input sources")
    a.output.mkdir(parents=True, exist_ok=False)
    m.TRAIN_REPLAY = a.train_replay
    m.SYNTHETIC = a.synthetic_cache
    needs_synthetic = a.mode == "recovery" or (a.mode == "measure" and not a.skip_calibration)
    if needs_synthetic and m.SYNTHETIC is None:
        m.SYNTHETIC = a.output / "synthetic.pkl"
        generate_aircraft_trajectories(save_path=str(m.SYNTHETIC))
    if a.mode == "recovery":
        with np.load(a.pool) as pool:
            result = m.recovery(pool, a.halves)
        (a.output / "recovery.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0
    days = m.train_days(a.raw)
    if a.mode == "day-ci":
        result = m.day_ci(days, a.pool_half, a.clip)
        (a.output / "day_ci.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0
    selected_days = days[:a.max_days or None]
    summary = dict(days=len(selected_days), halves=a.halves, clip=a.clip, gap=a.gap,
                   min_updates=a.min_updates, raw={}, calibration={})
    if not a.skip_calibration:
        for sigma, rho in ((0, 0), (5, 0), (5, .8), (25, 0), (25, .8)):
            for half in a.halves:
                residuals, pairs = m.decompose(m.synthetic(sigma, rho), half, a.clip)
                summary["calibration"][f"sigma{sigma}_rho{rho}_half{half:g}"] = m.stats(residuals["along"], pairs["along"])
    segments, intervals = m.load_segments(selected_days, a.gap, a.min_updates)
    summary.update(segments=len(segments), updates=int(sum(len(s[0]) for s in segments)),
                   update_interval_s=dict(median=float(np.median(intervals)), p10=float(np.percentile(intervals, 10)),
                                          p90=float(np.percentile(intervals, 90))))
    for half in a.halves:
        residuals, pairs = m.decompose(segments, half, a.clip)
        summary["raw"][f"half{half:g}"] = {k: m.stats(residuals[k], pairs[k]) for k in residuals}
        print(f"half width {half:g} complete", flush=True)
    pool_segments, _ = m.load_segments(days[:a.pool_days], a.gap, a.min_updates)
    residuals, _ = m.decompose(pool_segments, a.pool_half, a.clip)
    keep = (np.abs(residuals["along"]) < 500) & (np.abs(residuals["cross"]) < 500)
    np.savez(a.output / "residual_pool.npz", along=residuals["along"][keep], cross=residuals["cross"][keep])
    summary["pool"] = dict(file="residual_pool.npz", days=[Path(d).name[:8] for d in days[:a.pool_days]],
                           half=a.pool_half, n=int(keep.sum()), dropped_over_500m=int((~keep).sum()))
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

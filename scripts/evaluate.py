"""Evaluate released checkpoints and compare each seed with the saved paper results."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from merit.evaluation import predict
from merit.statistics import compare, kf_cv_forecast, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="prepared data directory containing raw_replay/")
    p.add_argument("--site", choices=["kbtp", "kagc"], default="kbtp")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--groups", nargs="+", default=["iso", "merit"])
    p.add_argument("--rows", nargs="+", default=["direct"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 0, 1])
    p.add_argument("--subsets", nargs="+", choices=["strict_unseen", "unseen", "all"], default=["strict_unseen", "unseen", "all"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=Path("outputs/evaluate"))
    p.add_argument("--target", default="merit direct")
    p.add_argument("--kf", nargs=2, type=float)
    p.add_argument("--smooth-reference", type=int, default=0)
    p.add_argument("--rtol", type=float, default=1e-6)
    a = p.parse_args()
    if not a.data.is_dir():
        raise ValueError("Prepared data directory does not exist")
    data_root = a.data.resolve()
    if a.output.resolve().is_relative_to(data_root):
        raise ValueError("Evaluation output must be outside the input data directory")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    filename = f"{a.split}.npz" if a.site == "kbtp" else f"kagc_{a.split}.npz"
    path = a.data / "raw_replay" / filename
    manifest = json.loads((ROOT / "merit/data_manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["raw_replay"][filename]["sha256"]:
        raise ValueError("Raw-replay cache differs from the paper build. Run prepare_data.py first.")
    with np.load(path) as z:
        data = dict(z)
    masks = {"strict_unseen": data["train_overlap_fraction"] < .1,
             "unseen": data["train_overlap_fraction"] <= .5,
             "all": np.ones(len(data["day"]), bool)}
    masks = {s: masks[s] for s in a.subsets}
    # Preserve the full inference batches even when reporting only one subset.
    history, reference, day = data["history_xy_m"], data["reference_xy_m"], data["day"]
    if a.smooth_reference:
        from scipy.signal import savgol_filter
        reference = savgol_filter(reference, a.smooth_reference, 2, axis=1, mode="interp")
    entries = json.loads((ROOT / "checkpoints/index.json").read_text())
    report = {"site": a.site, "split": a.split, "rows": {}, "comparisons": {}, "checks": []}
    errors_by_name = {}
    max_relative_difference = 0.0
    velocity = (history[:, -1] - history[:, -6]) / 5
    cv = history[:, -1, None] + velocity[:, None] * np.arange(1, 31)[None, :, None]
    errors_by_name["CV (last 5 s)"] = [np.linalg.norm(cv - reference, axis=-1)]
    if a.kf:
        name = f"KF-CV q={a.kf[0]:g} r={a.kf[1]:g}"
        errors_by_name[name] = [np.linalg.norm(kf_cv_forecast(history, *a.kf) - reference, axis=-1)]
    for group in a.groups:
        for row in a.rows:
            name = f"{group} {row}"
            errors_by_name[name] = []
            for seed in a.seeds:
                matches = [e for e in entries if (e["group"], e["row"], e["seed"]) == (group, row, seed)]
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one released checkpoint for {name}, seed {seed}")
                entry = matches[0]
                checkpoint = ROOT / "checkpoints" / entry["file"]
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Checkpoint hash mismatch for {checkpoint.name}")
                settings = entry["settings"]
                mean, scale = np.array(entry["mean"]), np.array(entry["scale"])
                prediction = predict(checkpoint, row, history, mean, scale, device=a.device,
                                     filter_cv=settings.get("filter_cv"),
                                     report_age=data["changed_position_age_s"] if settings.get("report_age") else None)
                error = np.linalg.norm(prediction - reference, axis=-1)
                errors_by_name[name].append(error)
                key = f"{a.site}_{a.split}"
                for subset, mask in masks.items():
                    observed = summarize(error, day, mask)
                    expected = entry["metrics"].get(key, {}).get(subset)
                    if expected is None or a.smooth_reference:
                        continue
                    for metric in ("ADE", "FDE", "day_equal_ADE"):
                        relative = abs(observed[metric] / expected[metric] - 1)
                        max_relative_difference = max(max_relative_difference, relative)
                        report["checks"].append({"group": group, "row": row, "seed": seed, "subset": subset,
                                                 "metric": metric, "relative_difference": relative})
                print(f"{name} seed {seed} complete", flush=True)
    for name, per_seed in errors_by_name.items():
        report["rows"][name] = {s: [summarize(e, day, mask) for e in per_seed] for s, mask in masks.items()}
    means = {n: np.mean(e, axis=0) for n, e in errors_by_name.items()}
    if a.target in means:
        report["comparisons"] = {n: {s: compare(means[a.target], e, day, mask) for s, mask in masks.items()}
                                 for n, e in means.items() if n != a.target}
        report["fde_comparisons"] = {n: {s: compare(means[a.target][:, -1:], e[:, -1:], day, mask)
                                          for s, mask in masks.items()} for n, e in means.items() if n != a.target}
    report["window_ADE_P95_m"] = {n: {s: float(np.percentile(e[mask].mean(1), 95)) for s, mask in masks.items()}
                                 for n, e in means.items()}
    report["max_relative_difference"] = max_relative_difference
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / f"{a.site}_{a.split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(a.output / f"{a.site}_{a.split}_errors.npz", names=np.array(list(means)),
                        errors=np.stack(list(means.values())).astype(np.float32), day=day,
                        train_overlap_fraction=data["train_overlap_fraction"],
                        mean_staleness_s=data["changed_position_age_s"].mean(1))
    print(f"Checked {len(report['checks'])} seed metrics. Maximum relative difference {max_relative_difference:.3g}")
    return 0 if max_relative_difference <= a.rtol and report["checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

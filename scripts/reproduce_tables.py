"""Recompute every printed numerical cell of Tables I, III, IV and V without source data.

Uses frozen measurement summaries and per-seed evaluation results. Standard deviations
use ddof=1. Printed confidence intervals are checked against the frozen bootstrap outputs.
Use --bootstrap to additionally regenerate paired CIs from the released window errors.
"""
import argparse
import json
import statistics
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PREDICTORS = ("direct", "projection", "gate_analytic", "patchtst", "transformer", "trajairnet")
MAIN = ["CV (last 5 s)", "KF-CV q=4 r=15", "iso direct", "gauss direct", "raw direct", "heur direct", "merit direct"]
MAIN += [f"{g} {r}" for r in PREDICTORS[1:] for g in ("iso", "merit")]
SITE2 = ["CV (last 5 s)", "KF-CV q=1 r=60", "iso direct", "heur direct", "meritkagc direct", "merit direct"]
SITE2 += [f"{g} {r}" for r in PREDICTORS[1:] for g in ("iso", "merit")]
VAL = [("raw_eval_val_p1", n) for n in ["iso100 direct", "gauss direct", "gauss+stale direct", "measured direct", "MERIT direct"]]
VAL += [("raw_eval_val_ablation", f"held_k{k} direct") for k in (10, 4, 1)]
VAL += [("raw_eval_val_p1", n) for n in ("raw_only direct", "MERIT_ft_raw direct")]
FACTOR = ["iso_matched direct", "iso_matched_stale direct", "measured direct", "merit direct"]
INPUT = ["iso direct", "ftp_iso direct", "ftp_merit direct", "merit direct", "iso_stale_age direct", "merit_age direct"]

# These are the confidence intervals quoted alongside the corresponding tables.
CI_SPECS = {
    "III LSTM": ("raw_eval_test_final", "comparisons", "iso direct", "strict_unseen"),
    "III TrajAirNet": ("raw_eval_test_sota", "comparisons", "iso trajairnet", "strict_unseen"),
    "III matched isotropic": ("raw_eval_test_final", "comparisons", "gauss direct", "strict_unseen"),
    "III raw training": ("raw_eval_test_final", "comparisons", "raw direct", "strict_unseen"),
    "III heuristic strict": ("raw_eval_test_heur", "comparisons", "heur direct", "strict_unseen"),
    "III heuristic all": ("raw_eval_test_heur", "comparisons", "heur direct", "all"),
    "IV LSTM": ("kagc_eval_test", "comparisons", "iso direct", "all"),
    "IV local KF": ("kagc_eval_test", "comparisons", "KF-CV q=1 r=60", "all"),
    "IV heuristic": ("kagc_eval_test", "comparisons", "heur direct", "all"),
    "IV local ages": ("kagc_eval_test", "comparisons", "meritkagc direct", "all"),
    "V staleness only": ("raw_eval_test_2x2", "comparisons", "iso_matched_stale direct", "strict_unseen"),
    "V structure only": ("raw_eval_test_2x2", "comparisons", "measured direct", "strict_unseen"),
    "V KAGC structure added": ("kagc_eval_test_2x2", "comparisons", "iso_matched_stale direct", "all"),
    "V filter iso KBTP": ("raw_eval_test_mb", "comparisons", "ftp_iso direct", "strict_unseen"),
    "V filter iso KAGC": ("kagc_eval_test_mb", "comparisons", "ftp_iso direct", "all"),
    "V filter MERIT": ("raw_eval_test_mb", "comparisons", "ftp_merit direct", "strict_unseen"),
    "V age KAGC": ("kagc_eval_test_mb", "comparisons", "merit_age direct", "all"),
    "V age residual structure": ("raw_eval_test_mb", "age_pair_comparisons_from_saved_errors", "ADE", "strict_unseen"),
}


def load(study):
    filename = "val.json" if "_val" in study else "test.json"
    return json.loads((ROOT / "results" / study / filename).read_text(encoding="utf-8"))


def q(value, places=1):
    return str(Decimal(repr(float(value))).quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP))


def metric(report, name, subset, key, sd=False):
    values = [r[key] for r in report["rows"][name][subset]]
    out = q(statistics.fmean(values))
    return out + "±" + q(statistics.stdev(values)) if sd and len(values) > 1 else out


def input_metric(report, name, subset):
    """Table V(c) was transcribed through a two-decimal experiment summary."""
    value = statistics.fmean(r["ADE"] for r in report["rows"][name][subset])
    return q(q(value, 2))


def measurement():
    folder = ROOT / "results/measurement"
    ci = json.loads((folder / "day_ci.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    kagc = json.loads((folder / "kagc_summary.json").read_text())
    out = {}
    for key in ("mad", "p95", "p99", "frac_over_50m"):
        row = []
        for component, places in (("along", 1), ("cross", 2)):
            v = ci[component][key]
            factor = 100 if key == "frac_over_50m" else 1
            if key == "frac_over_50m" and component == "cross":
                row.append("<0.01" if v["value"] * 100 < .01 else q(v["value"] * 100, 2))
            else:
                row.append(f"{q(v['value'] * factor, places)} [{q(v['ci95'][0] * factor, places)}, {q(v['ci95'][1] * factor, places)}]")
        out[key] = row
    out["acf1"] = [q(summary["raw"]["half10"][c]["acf_clipped"]["1"], 2) for c in ("along", "cross")]
    out["segments_updates"] = [str(summary["segments"]), str(summary["updates"])]
    out["kbtp_interval"] = [q(summary["update_interval_s"][k], 2) for k in ("median", "p10", "p90")]
    out["kagc_interval"] = [q(kagc["update_interval_s"][k], p) for k, p in (("median", 2), ("p10", 2), ("p90", 1))]
    for site, places in (("kbtp", (2, 2, 1)), ("kagc", (1, 1, 1))):
        with np.load(ROOT / f"merit/ages_{site}.npz") as z:
            ages = z["changed_position_age_s"]
        out[f"{site}_age"] = [q(v, p) for v, p in zip(np.percentile(ages, [50, 90, 99]), places)]
    v = ci["replay"]["repeat_fraction"]
    out["kbtp_repeat"] = [f"{q(v['value'] * 100)} [{q(v['ci95'][0] * 100)}, {q(v['ci95'][1] * 100)}]"]
    replay = json.loads((folder / "replay_statistics.json").read_text())
    out["kagc_repeat"] = [q(replay["kagc_repeat_fraction"] * 100, 0)]
    out["days"] = [str(summary["days"]), str(kagc["days"])]
    return out


def main_table(site):
    base = load("raw_eval_test_final" if site == "kbtp" else "kagc_eval_test")
    extra = load("raw_eval_test_heur") if site == "kbtp" else None
    sota = load("raw_eval_test_sota" if site == "kbtp" else "kagc_eval_test_sota")
    names = MAIN if site == "kbtp" else SITE2
    out = {}
    for name in names:
        report = sota if name.endswith(" trajairnet") else extra if name == "heur direct" and extra else base
        cells = []
        subsets = ("strict_unseen", "all") if site == "kbtp" else ("all",)
        for subset in subsets:
            cells += [metric(report, name, subset, "ADE", True), metric(report, name, subset, "FDE", site == "kbtp")]
            delta = ""
            if name.startswith("merit ") or name == "meritkagc direct":
                baseline = "iso " + name.split()[-1]
                a = statistics.fmean(r["ADE"] for r in report["rows"][name][subset])
                b = statistics.fmean(r["ADE"] for r in report["rows"][baseline][subset])
                change = (a / b - 1) * 100
                delta = ("+" if change > 0 else "") + q(change)
            elif name.startswith(("CV ", "KF-CV")):
                delta = "--"
            cells.append(delta)
        out[name] = cells
    return out


def components():
    out = {}
    for study, name in VAL:
        r = load(study)
        out["a " + name] = [metric(r, name, "strict_unseen", k) for k in ("ADE", "FDE")]
    for name in FACTOR:
        kb, ka = load("raw_eval_test_2x2"), load("kagc_eval_test_2x2")
        out["b " + name] = [metric(kb, name, s, "ADE") for s in ("strict_unseen", "all")]
        out["b " + name].append(metric(ka, name, "all", "ADE"))
    for name in INPUT:
        kb, ka = load("raw_eval_test_mb"), load("kagc_eval_test_mb")
        out["c " + name] = [input_metric(kb, name, s) for s in ("strict_unseen", "all")]
        out["c " + name] += [input_metric(ka, name, "all"), q(ka["window_ADE_P95_m"][name]["all"], 0)]
    return out


def confidence_intervals(bootstrap=False):
    from merit.statistics import compare
    out = {}
    for label, (study, block, name, subset) in CI_SPECS.items():
        r = load(study)
        comparison = r[block][name][subset]
        ci = comparison["ci95"]
        if bootstrap:
            with np.load(ROOT / "results" / study / "test_errors.npz") as z:
                names = list(z["names"])
                target = ("merit_age direct" if block == "age_pair_comparisons_from_saved_errors"
                          else "merit trajairnet" if name.endswith(" trajairnet") else "merit direct")
                baseline = "iso_stale_age direct" if block == "age_pair_comparisons_from_saved_errors" else name
                fraction = z["train_overlap_fraction"]
                mask = fraction < .1 if subset == "strict_unseen" else np.ones(len(fraction), bool)
                regenerated = compare(z["errors"][names.index(target)], z["errors"][names.index(baseline)], z["day"], mask)
                ci = regenerated["ci95"]
        out[label] = [q(q(v * 100, 2)) if study.endswith("_mb") else q(v * 100) for v in ci]
    return out


def recompute(bootstrap=False):
    return {"I": measurement(), "III": main_table("kbtp"), "IV": main_table("kagc"),
            "V": components(), "CI": confidence_intervals(bootstrap)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", action="store_true")
    a = parser.parse_args()
    computed = recompute(a.bootstrap)
    expected = json.loads((ROOT / "results/paper_tables.json").read_text(encoding="utf-8"))
    total = mismatches = 0
    for table, rows in expected.items():
        if set(computed[table]) != set(rows):
            raise ValueError(f"Row mismatch in table {table}")
        for label, cells in rows.items():
            got = computed[table][label]
            if len(got) != len(cells):
                raise ValueError(f"Column mismatch in table {table}, {label}")
            for i, (value, printed) in enumerate(zip(got, cells)):
                total += 1
                if value != printed:
                    mismatches += 1
                    print(f"{table} | {label} | column {i + 1} got {value!r}, paper {printed!r}")
    print(f"{total - mismatches} of {total} printed cells reproduced")
    return int(mismatches != 0)


if __name__ == "__main__":
    raise SystemExit(main())

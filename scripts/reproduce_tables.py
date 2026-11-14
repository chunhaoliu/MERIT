"""Recompute Tables I-V and the control results quoted in the text, without source data.

Uses frozen measurement summaries and per-seed evaluation results. Standard deviations
use ddof=1. Paired intervals use 2000 day-cluster draws, seed 0; --bootstrap also
regenerates the intervals available in the saved reports from released window errors.
"""
import argparse
import json
import statistics
import sys
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PREDICTORS = ("direct", "projection", "gate_analytic", "seq2seq", "wtftp", "transformer", "itransformer", "patchtst", "dlinear", "trajairnet")
MODERN = {"seq2seq": ("p10", "wtftp"), "wtftp": ("p13", "wtftp_w"), "itransformer": ("p10", "itransformer"), "patchtst": ("p10", "patchtst_r"), "dlinear": ("p10", "dlinear")}
MAIN = ["CV (last 5 s)", "KF-CV q=4 r=15"] + [f"{g} {r}" for r in PREDICTORS for g in ("iso", "merit")]
SITE2 = ["CV (last 5 s)", "KF-CV q=1 r=60"] + [f"{g} {r}" for r in PREDICTORS for g in ("iso", "merit")]

# These are the confidence intervals quoted alongside the corresponding tables.
CI_SPECS = {
    "matched isotropic KBTP strict": ("raw_eval_test_final", "merit direct", "gauss direct", "strict_unseen"),
    "raw training KBTP strict": ("raw_eval_test_final", "merit direct", "raw direct", "strict_unseen"),
    "heuristic KBTP strict": ("raw_eval_test_heur", "merit direct", "heur direct", "strict_unseen"),
    "heuristic KBTP all": ("raw_eval_test_heur", "merit direct", "heur direct", "all"),
    "local KF KAGC": ("kagc_eval_test", "merit direct", "KF-CV q=1 r=60", "all"),
    "heuristic KAGC": ("kagc_eval_test", "merit direct", "heur direct", "all"),
    "local ages KAGC": ("kagc_eval_test", "merit direct", "meritkagc direct", "all"),
    "filter iso KBTP strict": ("raw_eval_test_mb", "merit direct", "ftp_iso direct", "strict_unseen"),
    "filter iso KAGC": ("kagc_eval_test_mb", "merit direct", "ftp_iso direct", "all"),
    "filter MERIT KBTP strict": ("raw_eval_test_mb", "merit direct", "ftp_merit direct", "strict_unseen"),
    "age input KAGC": ("kagc_eval_test_mb", "merit direct", "merit_age direct", "all"),
    "age input residual structure KBTP strict": ("raw_eval_test_mb", "merit_age direct", "iso_stale_age direct", "strict_unseen"),
    "measured timing KBTP strict (three seeds)": ("raw_eval_test_2x2", "iso_matched_stale direct", "iso_matched direct", "strict_unseen"),
    "measured timing KAGC (three seeds)": ("kagc_eval_test_2x2", "iso_matched_stale direct", "iso_matched direct", "all"),
}


@lru_cache(maxsize=None)
def load(study):
    filename = "val.json" if "_val" in study else "test.json"
    if study.startswith("p10_"):
        site = study.split("_", 1)[1]; path = ROOT / "results/p10" / site / filename
    elif study.startswith("p13_"):
        site = study.split("_", 1)[1]; path = ROOT / "results/p13" / site / filename
    elif study.startswith("p12b_"):
        site = study.split("_", 1)[1]; path = ROOT / "results/p12b" / site / filename
    else:
        path = ROOT / "results" / study / filename
    return json.loads(path.read_text(encoding="utf-8"))


def source_for(display, site):
    if display in MODERN:
        family, source = MODERN[display]
        return f"{family}_{site}", source
    return ("raw_eval_test_sota" if site == "kbtp" and display == "trajairnet" else
            "kagc_eval_test_sota" if site == "kagc" and display == "trajairnet" else
            "raw_eval_test_final" if site == "kbtp" else "kagc_eval_test"), display


def q(value, places=1):
    return str(Decimal(repr(float(value))).quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP))


def metric(report, name, subset, key, sd=False):
    values = [r[key] for r in report["rows"][name][subset]]
    out = q(statistics.fmean(values))
    return out + "±" + q(statistics.stdev(values)) if sd and len(values) > 1 else out


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


def predictor_parameters():
    from merit.evaluation import make_model
    from merit.modern_models import ModernPredictor
    out = {}
    for name in PREDICTORS:
        if name in MODERN:
            source = MODERN[name][1]
            cfg = ({"hidden": 96, "enc_layers": 3, "dec_layers": 1} if source == "wtftp" else
                   {"hidden": 96, "enc_layers": 3, "dec_layers": 1, "maxlevel": 1} if source == "wtftp_w" else
                   {"d_model": 128, "heads": 4, "layers": 3, "ff": 256, "dropout": .1} if source == "itransformer" else
                   {"individual": True, "moving_avg": 25} if source == "dlinear" else
                   {"d_model": 128})
            model = ModernPredictor(source, np.zeros(6), np.ones(6), config=cfg)
        else:
            model = make_model(name, np.zeros(6), np.ones(6))
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        out[name] = [q(count / 1e6, 1 if name == "trajairnet" else 2)]
    return out


def comparison(study, target, baseline, subset, bootstrap=False):
    # Saved comparisons target the direct LSTM, except in the TrajAirNet study.
    report = load(study)
    if not bootstrap:
        if target == "merit direct" or (study.endswith("_sota") and target == "merit trajairnet"):
            saved = report.get("comparisons", {}).get(baseline, {}).get(subset)
            if saved is not None:
                return saved
        if target == "merit_age direct" and baseline == "iso_stale_age direct":
            return report["age_pair_comparisons_from_saved_errors"]["ADE"][subset]
    from merit.statistics import compare
    e, day, strict = errors(study)
    mask = strict if subset == "strict_unseen" else np.ones(len(day), bool)
    return compare(e[target], e[baseline], day, mask, draws=2000, seed=0)


def change_cells(c):
    return [signed(c["change"])] + [signed(v) for v in c["ci95"]]


def main_table(site, bootstrap=False):
    out = {}
    for display in PREDICTORS:
        study, source = source_for(display, site)
        report = load(study)
        subsets = ("strict_unseen", "all") if site == "kbtp" else ("all",)
        for group in ("iso", "merit"):
            name = f"{group} {display}"; source_name = f"{group} {source}"
            cells = []
            for subset in subsets:
                cells += [metric(report, source_name, subset, "ADE", True), metric(report, source_name, subset, "FDE", site == "kbtp")]
                if site == "kagc":
                    e, _, _ = errors(study); cells.append(q(np.percentile(e[source_name].mean(1), 95), 0))
                if group == "merit":
                    c = comparison(study, source_name, f"iso {source}", subset, bootstrap)
                    cells.append(f"{signed(c['change'])} [{signed(c['ci95'][0])}, {signed(c['ci95'][1])}]")
                else:
                    cells.append("")
            out[name] = cells
    # Preserve reference rows in their manuscript order.
    refs = {}
    for name in (("CV (last 5 s)", "KF-CV q=4 r=15") if site == "kbtp" else ("CV (last 5 s)", "KF-CV q=1 r=60")):
        report = load("raw_eval_test_final" if site == "kbtp" else "kagc_eval_test")
        cells = []
        for subset in (("strict_unseen", "all") if site == "kbtp" else ("all",)):
            cells += [metric(report, name, subset, "ADE"), metric(report, name, subset, "FDE", site == "kbtp")]
            if site == "kagc":
                e, _, _ = errors("kagc_eval_test"); cells.append(q(np.percentile(e[name].mean(1),95),0))
            cells.append("--")
        refs[name]=cells
    return {**refs, **out}


def components():
    # Seven residual/timing combinations, followed by five alternative remedies.
    specs = [
        ("Isotropic | no timing", "raw_eval_test_final", "iso direct", "kagc_eval_test", "iso direct", False, False),
        ("Matched isotropic | no timing", "raw_eval_test_final", "gauss direct", "kagc_eval_test_2x2", "iso_matched direct", False, True),
        ("Measured | no timing", "raw_eval_test_2x2", "measured direct", "kagc_eval_test_2x2", "measured direct", True, True),
        ("Near-zero | Bernoulli", "raw_eval_test_heur", "heur direct", "kagc_eval_test", "heur direct", False, False),
        ("Near-zero | measured ages", "p8_kbtp", "near0 direct", "p8_kagc", "near0 direct", False, False),
        ("Matched isotropic | measured ages", "p8_kbtp", "matched_stale direct", "p8_kagc", "matched_stale direct", False, False),
        ("Measured (MERIT) | measured ages", "raw_eval_test_final", "merit direct", "kagc_eval_test", "merit direct", False, False),
        ("Raw-data training", "raw_eval_test_final", "raw direct", None, None, False, False),
        ("Isotropic + Kalman front end", "raw_eval_test_mb", "ftp_iso direct", "kagc_eval_test_mb", "ftp_iso direct", False, False),
        ("MERIT + Kalman front end", "raw_eval_test_mb", "ftp_merit direct", "kagc_eval_test_mb", "ftp_merit direct", False, False),
        ("Matched isotropic + ages + age input", "raw_eval_test_mb", "iso_stale_age direct", "kagc_eval_test_mb", "iso_stale_age direct", False, False),
        ("MERIT + age input", "raw_eval_test_mb", "merit_age direct", "kagc_eval_test_mb", "merit_age direct", False, False),
    ]
    out = {}
    for label, kb, bn, ka, an, kb_three, ka_three in specs:
        cells = [metric(load(kb), bn, s, "ADE") + ("\u2020" if kb_three else "") for s in ("strict_unseen", "all")]
        if ka is None:
            cells += ["--", "--"]
        else:
            e, _, _ = errors(ka)
            mark = "\u2020" if ka_three else ""
            cells += [metric(load(ka), an, "all", "ADE") + mark, q(np.percentile(e[an].mean(1), 95), 0) + mark]
        out[label] = cells
    return out


def confidence_intervals(bootstrap=False):
    out = {}
    for label, (study, target, baseline, subset) in CI_SPECS.items():
        cells = change_cells(comparison(study, target, baseline, subset, bootstrap))
        out[label] = cells[1:] if label in ("local ages KAGC", "age input KAGC") else cells
    return out


@lru_cache(maxsize=None)
def errors(study):
    if study.startswith("p10_"):
        site = study.split("_", 1)[1]; path = ROOT / "results/p10" / site / "test_errors.npz"
    elif study.startswith("p13_"):
        site = study.split("_", 1)[1]; path = ROOT / "results/p13" / site / "test_errors.npz"
    else:
        path = ROOT / "results" / study / "test_errors.npz"
    with np.load(path) as z:
        return dict(zip(map(str, z["names"]), z["errors"])), z["day"], z["train_overlap_fraction"] < .1


def signed(value):
    return ("+" if value > 0 else "") + q(value * 100)


def text_results():
    """Date-disjoint retraining, five-seed timing controls and tail-error intervals quoted in the text."""
    from merit.statistics import compare, compare_p95
    out = {}

    def ade(label, study, target, other, strict=False, values=True, intervals=True):
        e, day, mask = errors(study)
        mask = mask if strict else np.ones(len(day), bool)
        c = compare(e[target], e[other], day, mask, draws=2000, seed=0)
        cells = [q(e[other][mask].mean()), q(e[target][mask].mean())] if values else []
        out[label] = cells + [signed(c["change"])] + ([signed(v) for v in c["ci95"]] if intervals else [])

    ade("date-disjoint KBTP all", "p8_dd_kbtp", "dd_merit direct", "dd_iso direct")
    ade("date-disjoint KBTP strict", "p8_dd_kbtp", "dd_merit direct", "dd_iso direct", True, values=False, intervals=False)
    ade("date-disjoint KAGC", "p8_dd_kagc", "dd_merit direct", "dd_iso direct", values=False, intervals=False)
    for name, group in (("near-zero residuals", "near0 direct"), ("matched isotropic", "matched_stale direct")):
        ade(f"{name} + ages KBTP strict", "p8_kbtp", group, "merit direct", True)
        ade(f"{name} + ages KAGC", "p8_kagc", group, "merit direct")
    for label, study, other in (("P95 isotropic", "kagc_eval_test", "iso direct"),
                                ("P95 local KF", "kagc_eval_test", "KF-CV q=1 r=60"),
                                ("P95 heuristic", "kagc_eval_test", "heur direct")):
        e, day, _ = errors(study)
        c = compare_p95(e["merit direct"], e[other], day, draws=2000, seed=0)
        out[label] = [q(c["p95"][1], 0), q(c["p95"][0], 0), signed(c["change"])] + [signed(v) for v in c["ci95"]]
    val = load("raw_eval_val_p1")
    out["validation residual/timing ablation"] = [metric(val, n + " direct", "strict_unseen", "ADE")
                                                for n in ("gauss", "MERIT")]
    out["validation raw fine-tuning"] = [metric(val, n + " direct", "strict_unseen", "ADE")
                                         for n in ("MERIT", "MERIT_ft_raw")]
    scale = load("raw_eval_val_ablation")
    values = [statistics.fmean(r["ADE"] for r in scale["rows"][f"held_k{k} direct"]["strict_unseen"]) for k in (1, 2, 10)]
    out["validation k bounds 1/2/10"] = [q(abs(values[0] - values[1])), q(values[2] - values[1])]
    out["KAGC local ages ADE"] = [metric(load("kagc_eval_test"), "meritkagc direct", "all", "ADE")]
    e, _, strict = errors("raw_eval_test_final")
    out["30 s horizon advantage"] = [q(e["iso direct"][strict, -1].mean() - e["merit direct"][strict, -1].mean(), 0)]
    with np.load(ROOT / "results/raw_eval_test_final/test_errors.npz") as z:
        age = z["mean_staleness_s"]
    edges = np.quantile(age, [0, .2, .4, .6, .8, .95, 1])
    cells = []
    for i in (0, 4, 5):
        mask = (age >= edges[i]) & (age <= edges[i + 1])
        gain = 100 * (1 - e["merit direct"][mask].mean(1).mean() / e["iso direct"][mask].mean(1).mean())
        cells += [q(np.median(age[mask]), 2), q(gain)]
    out["history-age gain"] = cells
    with np.load(ROOT / "results/figure_cases.npz") as cases:
        out["case mean ages"] = [q(v) for v in cases["changed_position_age_s"].mean(1)]
        out["case ADE isotropic to MERIT"] = [q(np.linalg.norm(cases[g + "_m"][i] - cases["reference_m"][i], axis=-1).mean(), 0)
                                              for i in range(3) for g in ("iso", "merit")]
    e, _, _ = errors("kagc_eval_test")
    out["KAGC median window ADE"] = [q(np.median(e[g + " direct"].mean(1)), 0) for g in ("iso", "merit")]
    processed = json.loads((ROOT / "results/processed_test.json").read_text())
    out["processed LSTM ADE/FDE"] = [q(statistics.fmean(r["test"]["0"][k] for r in processed if r["group"] == g and r["row"] == "direct"))
                                      for g in ("iso", "merit") for k in ("ADE", "FDE")]
    # Frozen age-bank and training-cost receipts used by the revised text.
    out["age-bank KBTP all ADE"] = [metric(load("p12b_kbtp"), f"n{n} direct", "all", "ADE") for n in (1, 5, 20, 65)]
    out["age-bank KAGC ADE"] = [metric(load("p12b_kagc"), f"n{n} direct", "all", "ADE") for n in (1, 5, 20, 65)]
    cost = json.loads((ROOT / "results/p11/summary.json").read_text())
    by_mode = {}
    for rec in cost["records"]:
        by_mode.setdefault(rec["mode"], []).extend(rec["seconds"][1:])
    out["training epoch cost"] = [q(statistics.fmean(by_mode["iso"]), 2), q(statistics.fmean(by_mode["merit"]), 2)]
    old_patch = load("raw_eval_test_final")
    out["original PatchTST strict ADE"] = [metric(old_patch, "iso patchtst", "strict_unseen", "ADE"), metric(old_patch, "merit patchtst", "strict_unseen", "ADE"), signed(-0.296)]
    return out


def recompute(bootstrap=False):
    return {"I": measurement(), "II": predictor_parameters(), "III": main_table("kbtp", bootstrap), "IV": main_table("kagc", bootstrap),
            "V": components(), "CI": confidence_intervals(bootstrap), "Text": text_results()}


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

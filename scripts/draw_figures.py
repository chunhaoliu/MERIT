"""Recreate Figures 2, 3 and 5 of MERIT from the released derived statistics.

Experimental panels are 2.30 by 1.82 inches with aligned physical plot areas.
Insert PDFs without width/scale options and typeset panel captions with subfig.
Cases reuse the original seed-42 selection rule without loading checkpoints.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.transforms import Bbox
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
# Okabe-Ito colors, with vermilion and a heavier stroke reserved for MERIT.
BLUE, MERIT, GREY = "#0072B2", "#D55E00", "#777777"
RESIDUAL = "#C00000"  # position residuals; blue is reserved for report timing
ISO, CROSS, REF = "#7E2F8E", "#009E73", "#E69F00"  # solid lines, separated by color
PAGE_WIDTH, GUTTER = 7.16, .07
EXPERIMENT_WIDTH, EXPERIMENT_HEIGHT = 2.30, 1.82


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix",
                         "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": .6,
                         "axes.edgecolor": "black", "axes.labelpad": 2,
                         "figure.facecolor": "white", "figure.frameon": False,
                         "axes.facecolor": "white", "axes.axisbelow": True,
                         "axes.grid": False,
                         "legend.frameon": True, "legend.fancybox": False, "legend.framealpha": 1,
                         "legend.edgecolor": "black", "patch.linewidth": .5, "legend.fontsize": 9,
                         "legend.handlelength": 1.8, "legend.handletextpad": .45,
                         "legend.labelspacing": .25, "legend.borderaxespad": .45,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
                         "xtick.major.pad": 2, "ytick.major.pad": 2,
                         "xtick.major.width": .6, "ytick.major.width": .6,
                         "xtick.minor.width": .4, "ytick.minor.width": .4})


def save(fig, out, name, bbox=None):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches=bbox, pad_inches=0)


def panel_row(left=.44):
    """Fixed physical rectangles leave room for ticks/labels within every PDF."""
    width = 3 * EXPERIMENT_WIDTH + 2 * GUTTER
    height = EXPERIMENT_HEIGHT
    fig = plt.figure(figsize=(width, height))
    axes = [fig.add_axes([(i * (EXPERIMENT_WIDTH + GUTTER) + left) / width,
                          .35 / height, (EXPERIMENT_WIDTH - left - .08) / width, 1.39 / height])
            for i in range(3)]
    return fig, axes


def save_panels(fig, out, name):
    fig.canvas.draw()
    height = fig.get_figheight()
    for i, letter in enumerate("abc"):
        # Exclude the other panels' invisible off-page text from each PDF.
        for j, ax in enumerate(fig.axes):
            ax.set_visible(i == j)
        left = i * (EXPERIMENT_WIDTH + GUTTER)
        save(fig, out, f"{name}_{letter}", Bbox.from_extents(left, 0, left + EXPERIMENT_WIDTH, height))
    plt.close(fig)
    (out / f"{name}_legend.pdf").unlink(missing_ok=True)


def error_structure(out):
    summary = json.loads((RESULTS / "measurement/summary.json").read_text())
    pool = np.load(ROOT / "merit/residual_pool.npz")
    age = np.load(ROOT / "merit/ages_kbtp.npz")["changed_position_age_s"].ravel()
    fig, axes = panel_row()
    ax = axes[0]
    bins = np.linspace(-150, 150, 301)
    for key, color, label in (("along", RESIDUAL, "Along-track"), ("cross", CROSS, "Cross-track")):
        r = pool[key]
        density, edges = np.histogram(r, bins=bins, density=True)
        visible = (edges[:-1] >= -100) & (edges[1:] <= 100)
        shown = np.flatnonzero(visible)
        ax.stairs(np.maximum(density[visible], 1e-5), edges[shown[0]:shown[-1] + 2], baseline=None,
                  color=color, linewidth=1.0, label=label)
    # Gaussian with the along-track scaled MAD: the measured tail lies far above it.
    r = pool["along"]; sd = 1.4826 * np.median(np.abs(r - np.median(r))); g = np.linspace(-100, 100, 401)
    ax.plot(g, np.exp(-.5 * (g / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), color=REF, linewidth=1.0,
            label="Gaussian, same MAD")
    ax.set_yscale("log"); ax.set_xlabel("Position residual (m)"); ax.set_ylabel("Density")
    ax.set_xlim(-100, 100); ax.set_ylim(1e-5, 1e4)
    ax.set_yticks([1e-4, 1e-2, 1, 1e2, 1e4])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext(base=10))
    ax.get_xticklabels()[0].set_ha("left")
    ax.get_xticklabels()[-1].set_ha("right")
    ax.legend(loc="upper left")
    ax = axes[1]
    lags = [1, 2, 3, 5, 10]
    measured = summary["raw"]["half10"]["along"]["acf_clipped"]
    ax.plot(lags, [measured[str(l)] for l in lags], color=RESIDUAL, linewidth=1.2, label="Measured along-track")
    for key, color, label in (("sigma25_rho0_half10", CROSS, "Synthetic white noise"),
                              ("sigma25_rho0.8_half10", REF, r"Synthetic AR(1), $\rho=0.8$")):
        c = summary["calibration"][key]["acf_clipped"]
        ax.plot(lags, [c[str(l)] for l in lags], color=color, linewidth=1.0, label=label)
    ax.set_xlabel("Lag (s)"); ax.set_ylabel("Residual autocorrelation"); ax.set_xticks(lags)
    ax.set_xlim(.5, 10.5); ax.set_ylim(-.82, 1.5); ax.set_yticks([-.5, 0, .5, 1])
    ax.legend(loc="upper right")
    ax = axes[2]
    kagc = np.load(ROOT / "merit/ages_kagc.npz")["changed_position_age_s"].ravel()
    for a, color, label in ((age, BLUE, "KBTP"), (kagc, REF, "KAGC")):
        x = np.sort(a)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=color, linewidth=1.1, label=label)
    ax.set_xscale("log"); ax.set_xlim(.05, 100); ax.set_ylim(0, 1.03); ax.set_xlabel("Displayed-position age (s)")
    ax.set_xticks([.1, 1, 10, 100], labels=["0.1", "1", "10", "100"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("Cumulative fraction")
    ax.get_xticklabels()[-1].set_ha("right")
    ax.legend(loc="lower right", handlelength=1.1)
    save_panels(fig, out, "fig_error")
    print("error c medians", np.median(age), np.median(kagc))


def results(out):
    z = np.load(RESULTS / "raw_eval_test_final/test_errors.npz")
    names = list(z["names"]); err = z["errors"]; strict = z["train_overlap_fraction"] < .1
    def get(n):
        return err[names.index(n)]

    fig, axes = panel_row()
    ax = axes[0]; h = np.arange(1, 31)
    iso, mer = get("iso direct")[strict].mean(0), get("merit direct")[strict].mean(0)
    for curve, label, color in ((iso, "LSTM, isotropic", ISO), (mer, "LSTM, MERIT", MERIT)):
        ax.plot(h, curve, color=color, linewidth=1.3 if color == MERIT else 1.0, label=label)
    ax.set_xlabel("Prediction horizon (s)"); ax.set_ylabel("Displacement error (m)")
    ax.set_xticks([0, 10, 20, 30]); ax.set_xlim(0, 31); ax.set_ylim(0, None)
    ax.legend(loc="upper left")

    # (b) Second site: fraction of windows whose ADE exceeds a level, so the tail is visible.
    k = np.load(RESULTS / "kagc_eval_test/test_errors.npz")
    kn = list(k["names"]); ax = axes[1]
    for n, label, color, lw in (([x for x in kn if x.startswith("KF-CV")][0], "KF-CV (local)", CROSS, 1.0),
                                ("iso direct", "LSTM, isotropic", ISO, 1.0),
                                ("merit direct", "LSTM, MERIT", MERIT, 1.3)):
        v = np.sort(k["errors"][kn.index(n)].mean(1))
        ax.step(v, 1 - np.arange(len(v)) / len(v), where="post", color=color, linewidth=lw, label=label)
        print("kagc P95", label, round(float(np.percentile(v, 95))))
    ax.axhline(.05, color="#aaaaaa", linewidth=.4, linestyle="--")
    ax.set_yscale("log"); ax.set_ylim(.01, 1.2); ax.set_xlim(0, 2400)
    ax.set_yticks([.01, .1, 1], labels=["0.01", "0.1", "1"])
    ax.set_xticks([0, 500, 1000, 1500, 2000], labels=["", "500", "1000", "1500", "2000"])
    ax.set_xlabel("Window ADE (m)"); ax.set_ylabel("Exceedance fraction")
    ax.legend(loc="upper right")

    ax = axes[2]; s = z["mean_staleness_s"]; day = z["day"]
    edges = np.quantile(s, [0, .2, .4, .6, .8, .95, 1])
    centers, gains, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s <= hi)
        centers.append(np.median(s[m])); counts.append(m.sum())
        gains.append([100 * (1 - get(f"merit {r}").mean(1)[m].mean() / get(f"iso {r}").mean(1)[m].mean())
                      for r in ("direct", "projection")])
    gains = np.array(gains)
    for c, (label, color, lw) in enumerate((("LSTM", MERIT, 1.3), ("LSTM+Fixed", ISO, 1.0))):
        ax.plot(centers, gains[:, c], color=color, linewidth=lw, label=label)
    ax.set_xscale("log"); ax.set_xlabel("Window-mean position age (s)"); ax.set_ylabel("ADE reduction (%)")
    ax.set_xticks([.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0, None); ax.legend(loc="lower right")
    save_panels(fig, out, "fig_results")
    print("staleness bins", np.round(centers, 2), "gain", np.round(gains[:, 0], 1), "n", counts)


def influence(out):
    """Column-width panels: residual spread versus range and speed, and per-aircraft timing spread."""
    data = json.loads((RESULTS / "measurement/influence.json").read_text(encoding="utf-8"))
    width, height, left, right, bottom, top = 3.45, 1.16, .54, .08, .34, .05
    for letter, key, scale, xlabel in (("a", "residual_range", 1e-3, "Range from receiver (km)"),
                                       ("b", "residual_speed", 1, "Ground speed (m/s)")):
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_axes([left / width, bottom / height, (width - left - right) / width, (height - bottom - top) / height])
        bins = data[key]["bins"]; x = np.array([b["median"] for b in bins]) * scale
        for comp, color, label in (("along", RESIDUAL, "Along-track"), ("cross", CROSS, "Cross-track")):
            ax.plot(x, [b[f"{comp}_mad"]["value"] for b in bins], color=color, linewidth=1.1, label=label)
        if letter == "b":
            k = np.median([b["along_mad"]["value"] / b["median"] for b in bins]); v = np.linspace(20, 100, 81)
            ax.plot(v, k * v, color=REF, linewidth=1.0, label=f"{k:.2f} s $\\times\\, v$")
            print("along MAD / speed", round(k, 3))
        ax.set_yscale("log"); ax.set_ylim(1, 200)
        ax.set_yticks([1, 5, 10, 50], labels=["1", "5", "10", "50"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(xlabel); ax.set_ylabel("MAD (m)")
        ax.set_xlim(*((20, 100) if letter == "b" else (0, 22)))
        if letter == "a":
            ax.set_xticks([0, 5, 10, 15, 20])
        else:
            ax.set_xticks([20, 40, 60, 80, 100]); ax.get_xticklabels()[-1].set_ha("right")
        ax.get_xticklabels()[0].set_ha("left")
        ax.legend(loc="upper left", ncol=2 if letter == "a" else 3, columnspacing=.9, handlelength=1.3)
        save(fig, out, f"fig_influence_{letter}")
        plt.close(fig)
        print("influence", letter, np.round(x, 1))
    stamps = json.loads((RESULTS / "measurement/timestamps.json").read_text(encoding="utf-8"))
    spread, split = np.array(stamps["aircraft_timing_mad_s"]), stamps["aircraft_split_s"]
    height = 1.25
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([left / width, bottom / height, (width - left - right) / width, (height - bottom - top) / height])
    bins = np.arange(.04, .2801, .01)
    ax.hist(spread[spread < split], bins=bins, color=REF, label=f"Below {split:.2f} s ({np.sum(spread < split)} aircraft)")
    ax.hist(spread[spread >= split], bins=bins, color=BLUE, label=f"Above {split:.2f} s ({np.sum(spread >= split)} aircraft)")
    ax.axvline(split, color=GREY, linestyle=":", linewidth=.9)
    ax.set_xlim(.04, .28); ax.set_xlabel(r"Per-aircraft timing spread, MAD$(e_\parallel/v)$ (s)"); ax.set_ylabel("Aircraft")
    ax.legend(loc="upper right")
    save(fig, out, "fig_influence_c")
    plt.close(fig)


def cases(out):
    """Strict-unseen windows (median-gain selection in the airborne stale decile): history age and horizon error."""
    data = np.load(RESULTS / "figure_cases.npz")
    case_meta = json.loads((RESULTS / "figure_cases.json").read_text())
    case_windows = case_meta.get("window_indices", [None] * 3)
    ref = data["reference_m"]
    pred = {"Isotropic": data["iso_m"], "MERIT": data["merit_m"]}
    age = data["changed_position_age_s"]
    stale = data["mean_age_s"]
    # Each panel: history axis (displayed-position age, t <= 0) beside horizon axis (error, t > 0).
    width = 3 * EXPERIMENT_WIDTH + 2 * GUTTER; height = EXPERIMENT_HEIGHT
    fig = plt.figure(figsize=(width, height))
    left, hist_w, gap, fut_w = .38, .80, .06, .66
    for i in range(3):
        x0 = i * (EXPERIMENT_WIDTH + GUTTER) + left
        ax_age = fig.add_axes([x0 / width, .35 / height, hist_w / width, 1.39 / height])
        ax = fig.add_axes([(x0 + hist_w + gap) / width, .35 / height, fut_w / width, 1.39 / height])
        ax_age.fill_between(np.arange(-59, 1), age[i], step="mid", color="#bdbdbd", linewidth=0)
        ax_age.set_xlim(-60, 0); ax_age.set_xticks([-60, -40, -20, 0]); ax_age.set_ylim(0, 1.1 * max(1, age[i].max()))
        ax_age.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True))
        ax_age.set_ylabel("Displayed age (s)")
        ax_age.set_xlabel("Time relative to prediction (s)")
        ax_age.xaxis.set_label_coords((hist_w + gap + fut_w) / 2 / hist_w, -.11)
        steps = np.arange(1, 31)
        for k, color, ls in (("Isotropic", BLUE, "--"), ("MERIT", MERIT, "-")):
            e = np.linalg.norm(pred[k][i] - ref[i], axis=-1)
            ax.plot(steps, e, ls, color=color, linewidth=1.4 if k == "MERIT" else 1.0,
                    label=k)
        ax.set_xlim(0, 30); ax.set_xticks([10, 20, 30]); ax.set_ylim(0, None)
        ax.tick_params(axis="y", labelleft=False, labelright=True)
        ax.yaxis.set_label_position("right"); ax.set_ylabel("Displacement error (m)")
        if i == 0:
            ax.legend(loc="upper left", handlelength=1.6)
        print(f"case {chr(97 + i)}: window {case_windows[i]}, mean age {stale[i]:.1f} s, max {age[i].max():.1f} s")
    fig.canvas.draw()
    for i, letter in enumerate("abc"):
        for j, a in enumerate(fig.axes):
            a.set_visible(j // 2 == i)
        left = i * (EXPERIMENT_WIDTH + GUTTER)
        save(fig, out, f"fig_cases_{letter}", Bbox.from_extents(left, 0, left + EXPERIMENT_WIDTH, height))
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/figures"), help="figure output directory")
    parser.add_argument("--only", nargs="+", choices=["error_structure", "influence", "results", "cases"], default=None)
    args = parser.parse_args()
    style()
    figures = {"error_structure": error_structure, "influence": influence,
               "results": results, "cases": cases}
    selected = args.only or ("error_structure", "influence", "results")
    for name, draw in figures.items():
        if name in selected:
            draw(args.output)
            print("drew", name)


if __name__ == "__main__":
    main()

"""Recreate Figures 1, 3 and 4 of MERIT from the released derived statistics.

Experimental panels are 2.30 by 1.82 inches with aligned physical plot areas.
Insert PDFs without width/scale options and typeset panel captions with subfig.
Runtime evidence is read-only; cases reuse the original seed-42 selection rule.
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
PAGE_WIDTH, GUTTER = 7.16, .07
EXPERIMENT_WIDTH, EXPERIMENT_HEIGHT = 2.30, 1.82


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix",
                         "font.size": 7.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": .6,
                         "axes.edgecolor": "black", "axes.labelpad": 2,
                         "figure.facecolor": "white", "figure.frameon": False,
                         "axes.facecolor": "white", "axes.axisbelow": True,
                         "axes.grid": False,
                         "legend.frameon": False, "legend.fontsize": 6.5,
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
    for key, color, ls, label in (("along", BLUE, "-", "Along-track"),
                                  ("cross", "#333333", "--", "Cross-track")):
        r = pool[key]
        density, edges = np.histogram(r, bins=bins, density=True)
        visible = (edges[:-1] >= -100) & (edges[1:] <= 100)
        shown = np.flatnonzero(visible)
        # Same histogram/density as the legacy plot, trimmed at its view limits.
        ax.stairs(np.maximum(density[visible], 1e-5), edges[shown[0]:shown[-1] + 2], baseline=None,
                  color=color, linestyle=ls, linewidth=1.0,
                  label=f"{label} (MAD {1.4826 * np.median(np.abs(r - np.median(r))):.1f} m)")
    ax.set_yscale("log"); ax.set_xlabel("Position residual (m)"); ax.set_ylabel("Density")
    ax.set_xlim(-100, 100); ax.set_ylim(1e-5, 3)
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=7))
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext(base=10))
    ax.tick_params(axis="y", labelsize=7.5)  # Log exponents remain above 5 pt.
    ax.get_xticklabels()[0].set_ha("left")
    ax.get_xticklabels()[-1].set_ha("right")
    ax.legend(loc="upper left")
    ax = axes[1]
    lags = [1, 2, 3, 5, 10]
    measured = summary["raw"]["half10"]["along"]["acf_clipped"]
    ax.plot(lags, [measured[str(l)] for l in lags], "o-", color=BLUE, markersize=3, linewidth=1.0, label="Measured along-track")
    for key, color, ls, marker, label in (("sigma25_rho0_half10", "#444444", "--", "s", "White, 25 m"),
                                   ("sigma25_rho0.8_half10", GREY, ":", "^", r"AR(1) $\rho$=0.8, 25 m")):
        c = summary["calibration"][key]["acf_clipped"]
        ax.plot(lags, [c[str(l)] for l in lags], ls, color=color, linewidth=1.0, marker=marker, markersize=3, label=label)
    ax.axhline(0, color="#aaaaaa", linewidth=.4, linestyle="--")
    ax.set_xlabel("Lag (s)"); ax.set_ylabel("Residual autocorrelation"); ax.set_xticks(lags)
    ax.set_xlim(.5, 10.5); ax.set_ylim(-.82, .56)
    ax.legend(loc="lower left")
    ax = axes[2]
    x = np.sort(age)
    ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=BLUE, linewidth=1.0)
    for q, ls in ((.5, "--"), (.9, ":")):
        v = np.quantile(age, q); ax.axvline(v, color=GREY, linestyle=ls, linewidth=.8)
        ax.annotate(f"P{int(q * 100)} = {v:.1f} s", (v, .06), xytext=(3, 0),
                    textcoords="offset points", rotation=90, va="bottom", color="#555555", fontsize=6.5)
    ax.set_xscale("log"); ax.set_xlim(.05, 60); ax.set_ylim(0, 1.03); ax.set_xlabel("Position age (s)")
    ax.set_xticks([.1, 1, 10], labels=["0.1", "1", "10"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(False, axis="x")
    ax.set_ylabel("Cumulative fraction")
    save_panels(fig, out, "fig_error")
    print(f"error c: {np.mean(age >= 1):.0%} of steps >= 1 s")


PREDICTORS = (("direct", "LSTM"), ("projection", "LSTM+Fixed"), ("gate_analytic", "LSTM+NARF"),
              ("patchtst", "PatchTST"), ("transformer", "Transformer"), ("trajairnet", "TrajAirNet"))


def results(out):
    z = np.load(RESULTS / "raw_eval_test_final/test_errors.npz")
    names = list(z["names"]); err = z["errors"]; strict = z["train_overlap_fraction"] < .1
    sota = np.load(RESULTS / "raw_eval_test_sota/test_errors.npz")
    # Only the new TrajAirNet row comes from SOTA; preserve all existing evidence.
    for key in ("day", "train_overlap_fraction", "mean_staleness_s"):
        np.testing.assert_array_equal(z[key], sota[key])

    def get(n):
        if n.endswith(" trajairnet"):
            return sota["errors"][list(sota["names"]).index(n)]
        return err[names.index(n)]

    fig, axes = panel_row(left=.63)
    ax = axes[0]; h = np.arange(1, 31)
    kf = [n for n in names if n.startswith("KF-CV")][0]
    # NPZ errors are already averaged over seeds; no per-horizon seed spread is stored.
    for n, label, color, ls, marker in ((kf, "KF-CV", "#555555", ":", "^"),
                                ("iso direct", "LSTM, isotropic", BLUE, "--", "s"),
                                ("merit direct", "LSTM, MERIT", MERIT, "-", "o")):
        ax.plot(h, get(n)[strict].mean(0), ls, color=color, linewidth=1.4 if color == MERIT else 1.0,
                marker=marker, markersize=2.6, markevery=[4, 9, 14, 19, 24, 29],
                markerfacecolor="white" if color != MERIT else color, markeredgewidth=.6, label=label)
    ax.set_xlabel("Prediction horizon (s)"); ax.set_ylabel("Displacement error (m)"); ax.set_xlim(1, 30); ax.set_ylim(0, None)
    ax.set_xticks([0, 10, 20, 30]); ax.set_xlim(0, 30)
    ax.legend(loc="upper left")
    ax = axes[1]; x = np.arange(len(PREDICTORS)); w = .34
    for j, (mask, label, face) in ((0, (strict, "Strict-unseen", MERIT)),
                                   (1, (np.ones_like(strict), "All windows", "#fae5d9"))):
        gain = [100 * (1 - get(f"merit {r}")[mask].mean() / get(f"iso {r}")[mask].mean()) for r, _ in PREDICTORS]
        bars = ax.barh(x + (j - .5) * .50, gain, w, color=face,
                       edgecolor=MERIT, label=label, linewidth=.45)
        for b, g in zip(bars, gain):
            ax.text(g + .7, b.get_y() + b.get_height() / 2, f"{g:.0f}", va="center", fontsize=6.5)
        print(label, "ADE reduction (%)", dict(zip([p for _, p in PREDICTORS], np.round(gain, 2))))
    ax.set_yticks(x, labels=[p for _, p in PREDICTORS]); ax.set_ylim(6.3, -.6)
    ax.set_xlabel("ADE reduction (%)"); ax.set_xlim(0, 36); ax.set_xticks([0, 10, 20, 30])
    ax.legend(loc="lower right", handlelength=1.1)
    ax = axes[2]; s = z["mean_staleness_s"]
    edges = np.quantile(s, [0, .2, .4, .6, .8, .95, 1])
    centers, gains, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s <= hi)
        centers.append(np.median(s[m])); counts.append(m.sum())
        gains.append([100 * (1 - get(f"merit {r}")[m].mean() / get(f"iso {r}")[m].mean()) for r in ("direct", "projection")])
    gains = np.array(gains)
    ax.plot(centers, gains[:, 0], "o-", color=MERIT, markersize=3, linewidth=1.4, label="LSTM")
    ax.plot(centers, gains[:, 1], "s--", color=BLUE, markersize=3, markerfacecolor="white",
            markeredgewidth=.6, linewidth=1.0, label="LSTM+Fixed")
    ax.set_xscale("log"); ax.set_xlabel("Median history age (s)"); ax.set_ylabel("ADE reduction (%)")
    ax.set_xticks([.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(6, 25); ax.legend(loc="lower right")
    save_panels(fig, out, "fig_results")
    print("staleness bins", np.round(centers, 2), "gain", np.round(gains[:, 0], 1), "n", counts)


def cases(out):
    """Strict-unseen windows: original median-gain selection in the airborne stale decile."""
    data = np.load(RESULTS / "figure_cases.npz")
    hist, ref = data["history_m"], data["reference_m"]
    pred = {"Isotropic": data["iso_m"], "MERIT": data["merit_m"]}
    ade = {"Isotropic": data["iso_ade_m"], "MERIT": data["merit_ade_m"]}
    stale = data["mean_age_s"]
    order = pick = np.arange(3)
    runs = ("Isotropic", "MERIT")
    fig, axes = panel_row()
    for i, ax in enumerate(axes):
        # Track frame at the prediction origin: along-track (km) versus cross-track (m),
        # so small lateral deviations stay visible on straight segments.
        origin = hist[i][-1]
        u = hist[i][-1] - hist[i][-11]; u = u / np.linalg.norm(u); n = np.array([-u[1], u[0]])
        zoom = i < 2  # Straight segments: magnified cross-track axis in meters; the turn stays to scale.
        frame = lambda q: np.stack([(q - origin) @ u / 1000, (q - origin) @ n / (1 if zoom else 1000)], -1)
        h, r = frame(hist[i][-20:]), frame(ref[i])
        ax.plot(h[:, 0], h[:, 1], "o", color=GREY, markersize=2, label="Raw history")
        ax.plot(r[:, 0], r[:, 1], "-", color="#222222", linewidth=1.0, label="Future reference")
        metrics, shown = [], [h[:, 1], r[:, 1]]
        for k, color, ls in (("Isotropic", BLUE, "--"), ("MERIT", MERIT, "-")):
            p = frame(pred[k][i]); shown.append(p[:, 1])
            e = np.linalg.norm(pred[k][i] - ref[i], axis=-1).mean()
            ax.plot(p[:, 0], p[:, 1], ls, color=color, linewidth=1.4 if k == "MERIT" else 1.0, label=k)
            metrics.append(f"{k} ADE {e:.0f} m")
        ax.plot(0, 0, "o", markersize=3.5, markerfacecolor="white", markeredgecolor="black", markeredgewidth=.7, zorder=5)
        ax.axhline(0, color="#bbbbbb", linewidth=.4, zorder=0)
        ax.set_xlabel("Along-track distance (km)")
        if zoom:
            lo, hi = min(map(np.min, shown)), max(map(np.max, shown)); pad = .25 * (hi - lo)
            ax.set_ylim(lo - pad - .9 * (hi - lo), hi + (.5 if i == 0 else .25) * (hi - lo)); ax.margins(x=.06)
            ax.set_ylabel("Cross-track offset (m)")
        else:
            ax.set_aspect("equal", adjustable="datalim"); ax.margins(.08)
            ax.set_ylabel("Cross-track offset (km)")
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.5))
        if i == 0:
            ax.legend(loc="lower left", ncol=2, columnspacing=.8, handlelength=1.6)
        ax.text((.96, .96, .04)[i], .95 if i == 0 else .05, "\n".join(f"{k}: {ade[k][order[i]]:.0f} m" for k in runs),
                transform=ax.transAxes, ha=("right", "right", "left")[i], va="top" if i == 0 else "bottom", fontsize=6.5, linespacing=1.25)
        print(f"case {chr(97 + i)}: Mean age {stale[pick[i]]:.1f} s; " + "; ".join(metrics), "window", int(pick[i]))
    save_panels(fig, out, "fig_cases")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/figures"), help="figure output directory")
    parser.add_argument("--only", nargs="+", choices=["error_structure", "results", "cases"], default=None)
    args = parser.parse_args()
    style()
    figures = {"error_structure": error_structure, "results": results, "cases": cases}
    for name, draw in figures.items():
        if args.only is None or name in args.only:
            draw(args.output)
            print("drew", name)


if __name__ == "__main__":
    main()

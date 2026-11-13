"""Recreate Figures 1, 3 and 4 of MERIT from the released derived statistics.

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
        ax.stairs(np.maximum(density[visible], 1e-5), edges[shown[0]:shown[-1] + 2], baseline=None,
                  color=color, linestyle=ls, linewidth=1.0, label=label)
    # Gaussian with the along-track scaled MAD: the measured tail lies far above it.
    r = pool["along"]; sd = 1.4826 * np.median(np.abs(r - np.median(r))); g = np.linspace(-100, 100, 401)
    ax.plot(g, np.exp(-.5 * (g / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), ":", color=GREY, linewidth=.9,
            label="Gaussian, along-track MAD")
    ax.set_yscale("log"); ax.set_xlabel("Position residual (m)"); ax.set_ylabel("Density")
    ax.set_xlim(-100, 100); ax.set_ylim(1e-5, 8)
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
    for key, color, ls, marker, label in (("sigma25_rho0_half10", "#444444", "--", "s", "Synthetic white noise"),
                                   ("sigma25_rho0.8_half10", GREY, ":", "^", r"Synthetic AR(1), $\rho=0.8$")):
        c = summary["calibration"][key]["acf_clipped"]
        ax.plot(lags, [c[str(l)] for l in lags], ls, color=color, linewidth=1.0, marker=marker, markersize=3, label=label)
    ax.axhline(0, color="#aaaaaa", linewidth=.4, linestyle="--")
    ax.set_xlabel("Lag (s)"); ax.set_ylabel("Residual autocorrelation"); ax.set_xticks(lags)
    ax.set_xlim(.5, 10.5); ax.set_ylim(-.82, .56)
    ax.legend(loc="lower left")
    ax = axes[2]
    kagc = np.load(ROOT / "merit/ages_kagc.npz")["changed_position_age_s"].ravel()
    for a, color, ls, lw, label in ((age, BLUE, "-", 1.2, "KBTP"), (kagc, "#333333", "--", 1.0, "KAGC")):
        x = np.sort(a)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), ls, color=color, linewidth=lw,
                label=f"{label} (median {np.median(a):.1f} s)")
    ax.set_xscale("log"); ax.set_xlim(.05, 100); ax.set_ylim(0, 1.03); ax.set_xlabel("Displayed-position age (s)")
    ax.set_xticks([.1, 1, 10, 100], labels=["0.1", "1", "10", "100"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("Cumulative fraction")
    ax.legend(loc="lower right")
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
    ax.fill_between(h, mer, iso, color=MERIT, alpha=.15, linewidth=0)
    for curve, label, color, ls, marker in ((iso, "LSTM, isotropic", BLUE, "--", "s"), (mer, "LSTM, MERIT", MERIT, "-", "o")):
        ax.plot(h, curve, ls, color=color, linewidth=1.4 if color == MERIT else 1.0,
                marker=marker, markersize=2.6, markevery=[4, 9, 14, 19, 24, 29],
                markerfacecolor="white" if color != MERIT else color, markeredgewidth=.6, label=label)
    ax.text(.97, .05, f"$-${iso[-1] - mer[-1]:.0f} m at 30 s", transform=ax.transAxes, ha="right", fontsize=6.5, color=MERIT)
    ax.set_xlabel("Prediction horizon (s)"); ax.set_ylabel("Displacement error (m)")
    ax.set_xticks([0, 10, 20, 30]); ax.set_xlim(0, 31); ax.set_ylim(0, None)
    ax.legend(loc="upper left")

    # (b) Second site: fraction of windows whose ADE exceeds a level, so the tail is visible.
    k = np.load(RESULTS / "kagc_eval_test/test_errors.npz")
    kn = list(k["names"]); ax = axes[1]
    for n, label, color, ls, lw in (([x for x in kn if x.startswith("KF-CV")][0], "KF-CV (KAGC-tuned)", "#555555", ":", 1.0),
                                    ("iso direct", "LSTM, isotropic", BLUE, "--", 1.0),
                                    ("merit direct", "LSTM, MERIT", MERIT, "-", 1.4)):
        v = np.sort(k["errors"][kn.index(n)].mean(1))
        ax.step(v, 1 - np.arange(len(v)) / len(v), ls, where="post", color=color, linewidth=lw, label=label)
        print("kagc P95", label, round(float(np.percentile(v, 95))))
    ax.axhline(.05, color="#aaaaaa", linewidth=.4, linestyle="--")
    ax.text(2350, .055, "P95", ha="right", va="bottom", fontsize=6.5, color="#555555")
    ax.set_yscale("log"); ax.set_ylim(.01, 1.2); ax.set_xlim(0, 2400)
    ax.set_yticks([.01, .1, 1], labels=["0.01", "0.1", "1"])
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_xlabel("Window ADE (m)"); ax.set_ylabel("Fraction of windows exceeding")
    ax.legend(loc="upper right")

    ax = axes[2]; s = z["mean_staleness_s"]; day = z["day"]
    edges = np.quantile(s, [0, .2, .4, .6, .8, .95, 1])
    rng = np.random.RandomState(0)
    centers, gains, bars, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s <= hi)
        centers.append(np.median(s[m])); counts.append(m.sum())
        row, ci = [], []
        groups = [np.flatnonzero(m & (day == d)) for d in np.unique(day[m])]
        draws = [np.concatenate([groups[g] for g in rng.randint(0, len(groups), len(groups))]) for _ in range(1000)]
        for r in ("direct", "projection"):
            a, b = get(f"merit {r}").mean(1), get(f"iso {r}").mean(1)
            row.append(100 * (1 - a[m].mean() / b[m].mean()))
            ci.append(np.percentile([100 * (1 - a[d].mean() / b[d].mean()) for d in draws], [2.5, 97.5]))
        gains.append(row); bars.append(ci)
    gains, bars = np.array(gains), np.array(bars)
    for c, (label, color, fmt, lw, face) in enumerate((("LSTM", MERIT, "o-", 1.4, MERIT), ("LSTM+Fixed", BLUE, "s--", 1.0, "white"))):
        ax.errorbar(centers, gains[:, c], yerr=[gains[:, c] - bars[:, c, 0], bars[:, c, 1] - gains[:, c]], fmt=fmt, color=color,
                    markersize=3, markerfacecolor=face, markeredgewidth=.6, linewidth=lw, elinewidth=.6, capsize=1.5, label=label)
    ax.set_xscale("log"); ax.set_xlabel("Median history age (s)"); ax.set_ylabel("ADE reduction (%)")
    ax.set_xticks([.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0, None); ax.legend(loc="lower right")
    save_panels(fig, out, "fig_results")
    print("staleness bins", np.round(centers, 2), "gain", np.round(gains[:, 0], 1), "ci", np.round(bars[:, 0], 1), "n", counts)


def cases(out):
    """Strict-unseen windows (median-gain selection in the airborne stale decile): history age and horizon error."""
    data = np.load(RESULTS / "figure_cases.npz")
    ref = data["reference_m"]
    pred = {"Isotropic": data["iso_m"], "MERIT": data["merit_m"]}
    age = data["changed_position_age_s"]
    stale = data["mean_age_s"]
    fig, axes = panel_row(left=.40)
    for i, ax in enumerate(axes):
        box = ax.get_position(); ax.set_position([box.x0, box.y0, box.width - .32 / fig.get_figwidth(), box.height])
        # History (t <= 0): age of the displayed position, the input the predictor receives.
        # Future (t > 0): displacement error of each predictor over the horizon.
        twin = ax.twinx()
        twin.fill_between(np.arange(-59, 1), age[i], step="mid", color="#cccccc", linewidth=0)
        twin.set_ylim(0, max(6, 1.15 * age[i].max()) * 2.2); twin.set_yticks([0, 2, 4, 6] if age[i].max() <= 6 else [0, 5, 10])
        twin.tick_params(axis="y", labelsize=6.5, colors="#555555", right=True)
        twin.set_ylabel("Displayed age (s)", color="#555555", fontsize=7); twin.yaxis.set_label_coords(1.2, .23)
        steps = np.arange(1, 31)
        for k, color, ls in (("Isotropic", BLUE, "--"), ("MERIT", MERIT, "-")):
            e = np.linalg.norm(pred[k][i] - ref[i], axis=-1)
            ax.plot(steps, e, ls, color=color, linewidth=1.4 if k == "MERIT" else 1.0,
                    label=k)
        ax.axvline(0, color="black", linewidth=.5)
        ax.set_zorder(twin.get_zorder() + 1); ax.patch.set_visible(False)
        ax.set_xlim(-60, 30); ax.set_xticks([-60, -40, -20, 0, 10, 20, 30]); ax.set_ylim(0, None)
        ax.tick_params(right=False)
        ax.set_xlabel("Time relative to prediction (s)"); ax.set_ylabel("Displacement error (m)")
        if i == 0:
            ax.legend(loc="upper left", handlelength=1.6)
        print(f"case {chr(97 + i)}: mean age {stale[i]:.1f} s, max {age[i].max():.1f} s")
    width, height = fig.get_figwidth(), fig.get_figheight()
    fig.canvas.draw()
    for i, letter in enumerate("abc"):
        for j in range(3):
            fig.axes[j].set_visible(i == j); fig.axes[3 + j].set_visible(i == j)
        left = i * (EXPERIMENT_WIDTH + GUTTER)
        save(fig, out, f"fig_cases_{letter}", Bbox.from_extents(left, 0, left + EXPERIMENT_WIDTH, height))
    plt.close(fig)



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

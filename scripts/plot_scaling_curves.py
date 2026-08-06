"""Plot the real-vs-imagined scaling analysis (random-init rotate-T).

Parses SCALE-EVAL-DONE lines from the sweep status logs (local + beast2 mirror),
aggregates per condition across seeds, and renders the user's requested figure:
x = #real demos, y = success; baseline (0 imagined) as a line; mixed-in imagined
doses as color-coded points at the same x (slight jitter); error bars across seeds.

Usage:
  python scripts/plot_scaling_curves.py \
      --logs <status.log> [<status2.log> ...] --out outputs/scaling_plot.png \
      [--min_seeds 1]
"""
import argparse
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# palette (reference instance, light mode)
SURF, INK, INK2, MUTED, GRID, BASE = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                      "#e1e0d9", "#c3c2b7")
BLUE = "#2a78d6"
# sequential aqua-family steps for imagined dose (light -> dark with dose)
DOSE_COLORS = {0: BLUE, 50: "#9fdcc4", 400: "#1baf7a", 800: "#0d7a52"}

LINE_RE = re.compile(
    r"SCALE-EVAL-DONE sc_r(\d+)i(\d+)_s(\d+): SUCCESS RATE:\s+([0-9.]+)%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/scaling_plot.png")
    ap.add_argument("--min_seeds", type=int, default=1)
    ap.add_argument("--title_suffix", default="")
    args = ap.parse_args()

    # condition -> {seed: rate}; later duplicates (reruns) overwrite earlier
    data = defaultdict(dict)
    for path in args.logs:
        with open(path, errors="ignore") as f:
            for line in f:
                m = LINE_RE.search(line)
                if m:
                    r, i, s, rate = (int(m.group(1)), int(m.group(2)),
                                     int(m.group(3)), float(m.group(4)))
                    data[(r, i)][s] = rate

    reals = sorted({k[0] for k in data})
    doses = sorted({k[1] for k in data})
    print("parsed conditions:")
    for (r, i), seeds in sorted(data.items()):
        print(f"  r{r} i{i}: " + ", ".join(f"s{s}={v:.0f}" for s, v in sorted(seeds.items())))

    fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=140)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=10.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    # baseline line (0 imagined)
    bx, by, be = [], [], []
    for r in reals:
        seeds = data.get((r, 0), {})
        if len(seeds) >= args.min_seeds:
            v = np.array(list(seeds.values()))
            bx.append(r)
            by.append(v.mean())
            be.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
    ax.errorbar(bx, by, yerr=be, color=BLUE, linewidth=2, marker="o", markersize=7,
                markerfacecolor=BLUE, markeredgecolor=SURF, capsize=4,
                label="real only (baseline)", zorder=3)

    # imagined-dose curves: one connected line per dose across the anchors,
    # points stacked directly on each anchor's x
    dose_order = [d for d in sorted(doses) if d != 0]
    for k, dose in enumerate(dose_order, start=1):
        xs, ys, es = [], [], []
        for r in reals:
            seeds = data.get((r, dose), {})
            if len(seeds) < args.min_seeds:
                continue
            v = np.array(list(seeds.values()))
            xs.append(r)
            ys.append(v.mean())
            es.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        if not xs:
            continue
        c = DOSE_COLORS.get(dose, "#1baf7a")
        ax.errorbar(xs, ys, yerr=es, color=c, linewidth=1.8, marker="o",
                    markersize=9, markerfacecolor=c, markeredgecolor=SURF,
                    markeredgewidth=1.2, capsize=4, zorder=4)
    # legend for doses
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BLUE, lw=2, marker="o", label="real only")]
    for dose in doses:
        if dose:
            handles.append(Line2D([], [], color=DOSE_COLORS.get(dose, "#1baf7a"),
                                  marker="o", linestyle="none", markersize=9,
                                  label=f"+{dose} imagined"))
    ax.legend(handles=handles, frameon=False, fontsize=10, labelcolor=INK2,
              loc="upper left")

    ax.axhline(54, color=MUTED, linewidth=1, linestyle=(0, (5, 3)), alpha=0.7)
    ax.annotate("800-real ceiling (~54%)", (max(reals) * 1.02, 55), color=MUTED,
                fontsize=9, ha="right")
    ax.set_xscale("log")
    ax.set_xticks(reals)
    ax.set_xticklabels([str(r) for r in reals])
    ax.minorticks_off()
    ax.set_xlabel("# real demos", color=INK2, fontsize=11.5)
    ax.set_ylabel("success rate (%)  —  n=50, seed 7000, final-ckpt rule", color=INK2,
                  fontsize=10.5)
    ax.set_title(
        f"Effect of real + imagined data mixtures{args.title_suffix}",
        color=INK, fontsize=13, loc="left", pad=12, fontweight="bold")
    ax.set_ylim(0, 72)
    plt.tight_layout()
    plt.savefig(args.out, facecolor=SURF, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

"""Plot DINO subject consistency + RAFT flow end-point error
(scripts/eval_wm_quality.py --mode video outputs).

Usage: python scripts/plot_wm_video_metrics.py --dir outputs/wm_quality \
           --out outputs/wm_video_metrics_plot.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURF, INK, INK2, MUTED, GRID, BASE = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                      "#e1e0d9", "#c3c2b7")
BLUE = "#2a78d6"
BLUE_LT = "#9ec4ea"
GREEN = "#1baf7a"
GREEN_LT = "#9fdcc4"


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def band(ax, v, color, label, ls="-"):
    t = np.arange(1, v.shape[1] + 1)
    ax.plot(t, np.nanmean(v, 0), color=color, lw=2, label=label, linestyle=ls)
    if ls == "-":
        ax.fill_between(t, np.nanpercentile(v, 25, 0), np.nanpercentile(v, 75, 0),
                        color=color, alpha=0.18, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/wm_quality")
    ap.add_argument("--out", default="outputs/wm_video_metrics_plot.png")
    args = ap.parse_args()
    d = np.load(Path(args.dir) / "video_metrics.npz")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4.8), dpi=140,
                                        gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    fig.patch.set_facecolor(SURF)

    style(ax0)
    band(ax0, d["aepe_color_rollout"], GREEN, "DP rollout in WM vs sim replay")
    band(ax0, d["aepe_color_teacher"], GREEN_LT, "teacher-forced vs real")
    ax0.set_xlabel("imagination step (10 Hz)", color=INK2, fontsize=10.5)
    ax0.set_ylabel("AEPE, normalized-RGB flow-map space  ↓", color=INK2,
                   fontsize=10.5)
    ax0.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")

    style(ax1)
    band(ax1, d["epe_rollout"], GREEN, "raw EPE: rollout vs sim replay")
    band(ax1, d["epe_teacher"], GREEN_LT, "raw EPE: teacher vs real")
    band(ax1, d["gtmag_rollout"], BLUE, "GT motion scale (sim |flow|)", ls="--")
    band(ax1, d["gtmag_teacher"], BLUE_LT, "GT motion scale (real |flow|)", ls="--")
    ax1.set_xlabel("imagination step (10 Hz)", color=INK2, fontsize=10.5)
    ax1.set_ylabel("raw flow end-point error (px, RAFT)  ↓", color=INK2,
                   fontsize=10.5)
    ax1.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")

    style(ax2)
    sets = [("real", "sc_real", BLUE), ("sim replay", "sc_sim_replay", BLUE_LT),
            ("teacher-\nforced WM", "sc_teacher_pred", GREEN_LT),
            ("WM rollout", "sc_wm_rollout", GREEN)]
    rng = np.random.default_rng(0)
    for i, (label, key, c) in enumerate(sets):
        v = d[key]
        x = np.full_like(v, i) + rng.uniform(-0.08, 0.08, size=len(v))
        ax2.scatter(x, v, s=22, color=c, alpha=0.55, edgecolors="none", zorder=3)
        ax2.hlines(v.mean(), i - 0.22, i + 0.22, color=c, lw=2.5, zorder=4)
        ax2.annotate(f"{v.mean():.3f}", (i, v.mean()), textcoords="offset points",
                     xytext=(0, -14), ha="center", color=INK2, fontsize=9)
    ax2.set_xticks(range(len(sets)))
    ax2.set_xticklabels([s[0] for s in sets], color=INK2, fontsize=9.5)
    ax2.set_ylabel("subject consistency (DINO)  ↑", color=INK2, fontsize=10.5)

    fig.suptitle("Imagined-video metrics: flow AEPE & subject consistency",
                 color=INK, fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.905, "24 episodes x 200 steps  ·  AEPE = per-pixel L2 between "
             "Middlebury color-wheel flow maps (per-frame normalized RGB)  ·  "
             "bands = IQR  ·  dots = videos, bar = mean", color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(args.out, facecolor=SURF, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

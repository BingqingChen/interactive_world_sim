"""Plot the world-model quality evaluation (scripts/eval_wm_quality.py outputs).

2x2 figure: PSNR / LPIPS / SSIM vs imagination step (paired rollout + teacher-
forced), and the T rotation-angle trajectories (WM-estimated vs ground-truth sim
replay). FID/KID numbers are printed in the figure footer.

Usage: python scripts/plot_wm_quality.py --dir outputs/wm_quality --out outputs/wm_quality_plot.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURF, INK, INK2, MUTED, GRID, BASE = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                      "#e1e0d9", "#c3c2b7")
BLUE = "#2a78d6"       # sim / real ground truth
GREEN = "#1baf7a"      # WM rollout (DP-in-WM, paired vs sim replay)
GREEN_LT = "#9fdcc4"   # WM teacher-forced (real actions vs real frames)


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def band(ax, v, color, label, start=0):
    """Mean line + IQR band across episodes for a (N, T) metric array. start=1
    skips frame 0, which is identical by construction (PSNR -> inf)."""
    t = np.arange(start, v.shape[1])
    v = v[:, start:]
    ax.plot(t, np.nanmean(v, 0), color=color, lw=2, label=label)
    ax.fill_between(t, np.nanpercentile(v, 25, 0), np.nanpercentile(v, 75, 0),
                    color=color, alpha=0.18, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/wm_quality")
    ap.add_argument("--out", default="outputs/wm_quality_plot.png")
    args = ap.parse_args()
    d = Path(args.dir)
    ro = np.load(d / "rollout_metrics.npz")
    te = np.load(d / "teacher_metrics.npz")
    fid = json.loads((d / "fid.json").read_text())

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=140)
    fig.patch.set_facecolor(SURF)

    ax = axes[0, 0]
    style(ax)
    band(ax, ro["psnr"], GREEN, "DP rollout in WM vs sim replay (paired)", start=1)
    band(ax, te["psnr"], GREEN_LT, "teacher-forced (real actions) vs real frames", start=1)
    ax.set_ylabel("PSNR (dB)  ↑", color=INK2, fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")

    ax = axes[0, 1]
    style(ax)
    band(ax, ro["lpips"], GREEN, None, start=1)
    band(ax, te["lpips"], GREEN_LT, None, start=1)
    ax.set_ylabel("LPIPS (AlexNet)  ↓", color=INK2, fontsize=10.5)

    ax = axes[1, 0]
    style(ax)
    band(ax, ro["ssim"], GREEN, None, start=1)
    band(ax, te["ssim"], GREEN_LT, None, start=1)
    ax.set_ylabel("SSIM  ↑", color=INK2, fontsize=10.5)
    ax.set_xlabel("imagination step (10 Hz)", color=INK2, fontsize=10.5)

    ax = axes[1, 1]
    style(ax)
    wm_deg = -np.degrees(ro["wm_angle"])   # CW positive
    gt_deg = -np.degrees(ro["gt_angle"])
    band(ax, wm_deg, GREEN, "T angle in WM (estimator)")
    band(ax, gt_deg, BLUE, "T angle in sim replay (ground truth)")
    ax.axhline(80, color=MUTED, lw=1, linestyle=(0, (5, 3)), alpha=0.7)
    ax.annotate("80° CW terminal", (wm_deg.shape[1] * 0.62, 82), color=MUTED,
                fontsize=8.5)
    ax.set_ylim(top=95)
    ax.set_ylabel("T rotation (CW deg)", color=INK2, fontsize=10.5)
    ax.set_xlabel("imagination step (10 Hz)", color=INK2, fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")

    n, L = ro["psnr"].shape
    fig.suptitle("World-model imagined rollouts vs ground-truth simulator",
                 color=INK, fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.945,
             f"paired: {n} episodes x {L} steps, identical initial state + identical "
             f"actions (random-init protocol)  ·  bands = IQR across episodes",
             color=INK2, fontsize=9.5)
    footer = (f"distributional (10k frames, InceptionV3): "
              f"FID imagined vs real = {fid['fid_imagined_vs_real']:.1f} "
              f"(real-vs-real floor {fid['fid_real_floor']:.1f})   ·   "
              f"KID×1000 = {fid['kid_imagined_vs_real']:.1f}±"
              f"{fid['kid_imagined_vs_real_std']:.1f} "
              f"(floor {fid['kid_real_floor']:.1f})")
    y0 = 0.005
    fp_path = d / "fid_paired.json"
    if fp_path.exists():
        fp = json.loads(fp_path.read_text())
        fig.text(0.02, 0.005,
                 f"paired sets ({fp['n_samples']} frames): "
                 f"FID rollout WM vs sim replay = {fp['fid_rollout_wm_vs_simreplay']:.1f}"
                 f"   ·   teacher pred vs real = {fp['fid_teacher_pred_vs_real']:.1f}"
                 f"   ·   sim replay vs real pool = {fp['fid_simreplay_vs_realpool']:.1f}"
                 f"   (matched-n floor {fp['fid_real_floor_matched_n']:.1f})",
                 color=INK2, fontsize=9.5)
        y0 = 0.028
    fig.text(0.02, y0, footer, color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.02 + y0, 1, 0.93))
    fig.savefig(args.out, facecolor=SURF, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot per-window uncertainty predictors vs. realized rollout PSNR.

Reads the CSV produced by scripts/eval_metric_correlation.py and draws one
scatter panel per predictor (inter-seed latent variance, round-trip residual),
with the absolute rollout PSNR on the y-axis. Each point is one window; the
overlaid curve is the per-bin median over quantile bins; the pooled Spearman
rho is annotated on each panel and printed to the console.

Layout and metrics follow mmbench2's Figure 5, minus the normalizations
(absolute PSNR instead of a repeated-frame-baseline delta, raw predictor
values instead of scene-motion-normalized ones). Styled with plain matplotlib.

Usage:
    python scripts/plot_metric_correlation.py \\
        --csv outputs/metric_correlation/window_metrics.csv \\
        --out outputs/metric_correlation/metric_correlation
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

POINT_COLOR = "#9150BE"
CURVE_COLOR = "#75409C"

Y_COLUMN = "psnr_db"
Y_LABEL = "Rollout PSNR (dB)"

# Human-readable labels for known predictor columns. Any column present in the CSV but not
# listed here (e.g. new *_ef_*_mean consistency columns) is auto-labelled from its name.
COLUMN_LABELS = {
    "interseed_var": "Inter-seed latent variance",
    "roundtrip_residual": "Round-trip residual",
    "dino_ef_openloop_mean": "DINO $e_F$ (open-loop)",
    "dino_ef_reanchor_mean": "DINO $e_F$ (re-anchor)",
    "vjepa_ef_openloop_mean": "V-JEPA $e_F$ (open-loop)",
    "vjepa_ef_reanchor_mean": "V-JEPA $e_F$ (re-anchor)",
}


def build_panels(df: pd.DataFrame, columns: list[str] | None) -> list[dict]:
    """Panels to plot: an explicit --columns list, else every known/`*_ef_*_mean` predictor."""
    if columns:
        cols = [c for c in columns if c in df.columns]
    else:
        cols = [c for c in COLUMN_LABELS if c in df.columns]
        cols += [c for c in df.columns if c.endswith("_ef_mean") or c.endswith("_ef_openloop_mean")
                 or c.endswith("_ef_reanchor_mean")]
        cols = list(dict.fromkeys(cols))  # de-dup, preserve order
    return [{"x": c, "x_label": COLUMN_LABELS.get(c, c.replace("_", " "))} for c in cols]

RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelcolor": "0.10",
    "axes.edgecolor": "0.30",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.color": "0.20",
    "ytick.color": "0.20",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def quantile_calibration(
    x: np.ndarray, y: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Median of y within quantile bins of x -> (bin_x_medians, bin_y_medians)."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < n_bins * 3:
        return np.array([]), np.array([])
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(x, edges, right=False) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() < 3:
            continue
        xs.append(np.median(x[mask]))
        ys.append(np.median(y[mask]))
    return np.array(xs), np.array(ys)


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pooled Spearman rho and p-value over finite pairs."""
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 5:
        return float("nan"), float("nan")
    rho, p = spearmanr(x[finite], y[finite])
    return float(rho), float(p)


def make_plot(df: pd.DataFrame, out_stem: Path, n_bins: int, panels: list[dict]) -> None:
    plt.rcParams.update(RC_PARAMS)
    ncols = min(len(panels), 3)
    nrows = -(-len(panels) // ncols)  # ceil
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.6 * ncols, 2.3 * nrows), dpi=200, sharey=True, squeeze=False
    )
    flat_axes = axes.flatten()

    for i, (ax, panel) in enumerate(zip(flat_axes, panels)):
        x = df[panel["x"]].to_numpy(dtype=float)
        y = df[Y_COLUMN].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)

        ax.scatter(
            x[finite],
            y[finite],
            s=7,
            alpha=0.25,
            color=POINT_COLOR,
            linewidth=0,
            zorder=2,
            rasterized=True,
        )

        # Per-quantile-bin median curve with a white halo for legibility.
        xb, yb = quantile_calibration(x, y, n_bins=n_bins)
        if xb.size > 0:
            ax.plot(xb, yb, color="white", linewidth=3.2, zorder=4, solid_capstyle="round")
            ax.plot(xb, yb, color=CURVE_COLOR, linewidth=1.8, zorder=5, solid_capstyle="round")
            ax.plot(xb, yb, "o", color=CURVE_COLOR, markersize=3.3, markeredgewidth=0, zorder=6)

        rho, _ = spearman(x, y)
        ax.text(
            0.97,
            0.96,
            rf"$\rho = {rho:+.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
            color="0.05",
            zorder=7,
        )

        ax.set_xlabel(panel["x_label"], labelpad=3)
        if i % ncols == 0:
            ax.set_ylabel(Y_LABEL, labelpad=3)
        ax.grid(True, which="major", axis="y", color="0.93", linewidth=0.5, zorder=0)
        ax.tick_params(axis="both", which="major", pad=2)

    for ax in flat_axes[len(panels):]:  # hide unused axes
        ax.set_visible(False)

    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
    print(f"Wrote {out_stem.with_suffix('.png')}")

    print(f"\nPer-window Spearman rho (n={len(df)}):")
    for panel in panels:
        rho, p = spearman(
            df[panel["x"]].to_numpy(dtype=float), df[Y_COLUMN].to_numpy(dtype=float)
        )
        print(f"  {panel['x']:<24s} vs {Y_COLUMN:<10s} rho={rho:+.3f}  p={p:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("outputs/metric_correlation/window_metrics.csv")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/metric_correlation/metric_correlation"),
        help="Output path stem; .png and .pdf are appended",
    )
    parser.add_argument("--n_bins", type=int, default=10, help="Quantile bins for the median curve")
    parser.add_argument(
        "--columns",
        default=None,
        help="Comma-separated predictor columns to plot (default: all known + *_ef_*_mean)",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"Missing {args.csv} — run scripts/eval_metric_correlation.py first")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} windows from {args.csv}")
    columns = args.columns.split(",") if args.columns else None
    panels = build_panels(df, columns)
    if not panels:
        raise ValueError(f"No predictor columns found in {args.csv}")
    make_plot(df, args.out, n_bins=args.n_bins, panels=panels)


if __name__ == "__main__":
    main()

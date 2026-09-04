"""Stage C1 of the H1/H2 study: merge the paired-collection metadata and the
action-distribution scores into one per-window table, and produce the
distribution-review material. NO training data is built here -- the split rule,
AD-axis metric, and budget B are decided by the user at this review; C2 consumes
the table afterwards.

Inputs per shard (from collect_imagined_rotate_t_paired.py + score_chunk_ad.py):
  pair<S>_meta.npz / .json, pair<S>_ad.npz / .json, the two frame zarrs.

Outputs (all under --out_dir):
  window_table.npz  one row per trainable hallucinated window:
      shard, episode (shard-local), t (plan-start frame), drift (t mod R),
      psnr_obs (mean over the 2 obs frames -- what training actually sees),
      psnr_plan (mean over the plan's 8 generated frames), psnr_win (mean over
      the 16-frame window), ssim/lpips left for later (frames are in the zarrs),
      mse, mse_exec, kl, kl_floor, decoder_floor (episode-cycle mean),
      angle_progress (sim T-angle change over the plan).
  review.json       medians, per-drift-level PSNR ladder, Spearman correlations,
      quadrant occupancy for median splits on (psnr_obs x kl) and (psnr_obs x mse).
  plots/*.png       psnr-by-drift ladder, metric histograms, psnr-vs-AD scatters.
  gallery/*.png     W|S side-by-side strips sampled per drift level and per
      psnr_obs quantile (visual "what does the data look like").

Usage:
  python scripts/build_h1h2_review.py --shards D,E,F,G [--pair_prefix datasets/]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score_chunk_ad import enumerate_windows  # noqa: E402

DP_ROOT_CANDIDATES = [
    "/home/jacobhb/projects/worth_doing/diffusion_policy",
    "/home/jacob/projects/worth_doing/diffusion_policy",
]
for p in DP_ROOT_CANDIDATES:
    if Path(p).exists():
        sys.path.insert(0, p)
        break
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


def load_shard(pre, shard):
    meta = np.load(pre / f"pair{shard}_meta.npz")
    man = json.loads((pre / f"pair{shard}_meta.json").read_text())
    ad_path = pre / f"pair{shard}_ad.npz"
    ad = np.load(ad_path) if ad_path.exists() else None
    return meta, man, ad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="D,E,F,G")
    ap.add_argument("--pair_prefix", default="datasets/")
    ap.add_argument("--out_dir", default="outputs/h1h2/review")
    ap.add_argument("--gallery_per_cell", type=int, default=3)
    args = ap.parse_args()

    pre = Path(args.pair_prefix)
    out = Path(args.out_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "gallery").mkdir(parents=True, exist_ok=True)

    cols = {k: [] for k in ("shard", "episode", "t", "drift", "psnr_obs",
                            "psnr_plan", "psnr_win", "mse", "mse_exec", "kl",
                            "kl_floor", "decoder_floor", "angle_progress")}
    R = None
    for shard in args.shards.split(","):
        meta, man, ad = load_shard(pre, shard)
        R = man["resync_period"]
        starts = np.r_[0, meta["episode_ends"][:-1]]
        ad_map = {}
        if ad is not None:
            ad_map = {(int(e), int(t)): i
                      for i, (e, t) in enumerate(zip(ad["episode"], ad["t"]))}
        fc = meta["decoder_floor_counts"]
        fstarts = np.r_[0, np.cumsum(fc)[:-1]]
        for ep, (s, L) in enumerate(zip(starts, meta["ep_len"])):
            psnr = meta["psnr"][s:s + L]
            ang = meta["gt_angle"][s:s + L]
            dfloor = float(np.nanmean(
                meta["decoder_floor"][fstarts[ep]:fstarts[ep] + fc[ep]]))
            for t in enumerate_windows(int(L), R):
                i = ad_map.get((ep, t))
                cols["shard"].append(shard)
                cols["episode"].append(ep)
                cols["t"].append(t)
                cols["drift"].append(t % R)
                cols["psnr_obs"].append(float(np.mean(psnr[t - 1:t + 1])))
                cols["psnr_plan"].append(float(np.mean(psnr[t + 1:t + 9])))
                cols["psnr_win"].append(float(np.mean(psnr[t - 1:t + 15])))
                cols["mse"].append(float(ad["mse"][i]) if i is not None else np.nan)
                cols["mse_exec"].append(float(ad["mse_exec"][i]) if i is not None else np.nan)
                cols["kl"].append(float(ad["kl"][i]) if i is not None else np.nan)
                cols["kl_floor"].append(float(ad["kl_floor"][i]) if i is not None else np.nan)
                cols["decoder_floor"].append(dfloor)
                cols["angle_progress"].append(float(np.degrees(ang[min(t + 8, L - 1)] - ang[t])))

    tab = {k: np.array(v) for k, v in cols.items()}
    np.savez_compressed(out / "window_table.npz", **tab)
    n = len(tab["t"])

    # ---- stats ----
    from scipy.stats import spearmanr
    fin = np.isfinite(tab["kl"])
    drift_levels = sorted(set(tab["drift"].tolist()))
    ladder = {int(d): dict(
        n=int((tab["drift"] == d).sum()),
        psnr_obs_median=float(np.median(tab["psnr_obs"][tab["drift"] == d])),
        kl_median=float(np.nanmedian(tab["kl"][tab["drift"] == d])),
    ) for d in drift_levels}

    def quad(x, y):
        mx, my = np.nanmedian(x), np.nanmedian(y)
        return {"hi_hi": int(((x > mx) & (y > my)).sum()),
                "hi_lo": int(((x > mx) & (y <= my)).sum()),
                "lo_hi": int(((x <= mx) & (y > my)).sum()),
                "lo_lo": int(((x <= mx) & (y <= my)).sum()),
                "x_median": float(mx), "y_median": float(my)}

    review = dict(
        n_windows=n, R=R, shards=args.shards,
        n_scored=int(fin.sum()),
        psnr_obs=dict(median=float(np.median(tab["psnr_obs"])),
                      p10=float(np.percentile(tab["psnr_obs"], 10)),
                      p90=float(np.percentile(tab["psnr_obs"], 90))),
        decoder_floor_median=float(np.nanmedian(tab["decoder_floor"])),
        kl=dict(median=float(np.nanmedian(tab["kl"])),
                floor_median=float(np.nanmedian(tab["kl_floor"]))),
        mse_median=float(np.nanmedian(tab["mse"])),
        drift_ladder=ladder,
        spearman=dict(
            mse_kl=float(spearmanr(tab["mse"][fin], tab["kl"][fin]).statistic),
            psnr_kl=float(spearmanr(tab["psnr_obs"][fin], tab["kl"][fin]).statistic),
            psnr_mse=float(spearmanr(tab["psnr_obs"][fin], tab["mse"][fin]).statistic),
            drift_psnr=float(spearmanr(tab["drift"], tab["psnr_obs"]).statistic),
        ),
        quadrants_psnr_x_kl=quad(tab["psnr_obs"][fin], tab["kl"][fin]),
        quadrants_psnr_x_mse=quad(tab["psnr_obs"][fin], tab["mse"][fin]),
    )
    (out / "review.json").write_text(json.dumps(review, indent=2))
    print(json.dumps(review, indent=2))

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.violinplot([tab["psnr_obs"][tab["drift"] == d] for d in drift_levels],
                  positions=drift_levels, widths=6)
    ax.axhline(review["decoder_floor_median"], ls="--", c="gray", label="decoder floor")
    ax.set_xlabel("obs drift (frames since resync)"); ax.set_ylabel("psnr_obs (dB)")
    ax.set_title("fidelity ladder"); ax.legend()
    ax = axes[0, 1]
    ax.hist(tab["kl"][fin], bins=60, alpha=0.7, label="KL(p||q)")
    fl = tab["kl_floor"][np.isfinite(tab["kl_floor"])]
    if len(fl):
        ax.hist(fl, bins=60, alpha=0.7, label="floor p||p'")
    ax.set_xlabel("nats"); ax.set_title("forward KL"); ax.legend()
    ax = axes[1, 0]
    ax.scatter(tab["psnr_obs"][fin], tab["kl"][fin], s=3, alpha=0.3)
    ax.axvline(review["quadrants_psnr_x_kl"]["x_median"], c="r", lw=0.8)
    ax.axhline(review["quadrants_psnr_x_kl"]["y_median"], c="r", lw=0.8)
    ax.set_xlabel("psnr_obs (dB)"); ax.set_ylabel("KL (nats)")
    ax.set_title(f"rho={review['spearman']['psnr_kl']:.2f}")
    ax = axes[1, 1]
    ax.scatter(tab["psnr_obs"][fin], tab["mse"][fin], s=3, alpha=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("psnr_obs (dB)"); ax.set_ylabel("action MSE (m^2)")
    ax.set_title(f"rho={review['spearman']['psnr_mse']:.2f}")
    fig.tight_layout()
    fig.savefig(out / "plots" / "review.png", dpi=140)

    # ---- gallery: W|S strips by drift level x psnr tercile ----
    import cv2
    rng = np.random.default_rng(0)
    zarrs = {}
    for shard in args.shards.split(","):
        zarrs[shard] = (
            ReplayBuffer.copy_from_path(
                str(pre / f"rotate_t_imagined_pair{shard}_dp.zarr"), keys=["img"]),
            ReplayBuffer.copy_from_path(
                str(pre / f"rotate_t_simreplay_pair{shard}_dp.zarr"), keys=["img"]))
    terc = np.nanpercentile(tab["psnr_obs"], [33, 66])
    for d in drift_levels:
        for ti, (lo, hi) in enumerate([(-np.inf, terc[0]), (terc[0], terc[1]),
                                       (terc[1], np.inf)]):
            sel = np.where((tab["drift"] == d) & (tab["psnr_obs"] > lo)
                           & (tab["psnr_obs"] <= hi))[0]
            if not len(sel):
                continue
            rows_img = []
            for i in rng.choice(sel, min(args.gallery_per_cell, len(sel)), replace=False):
                sh, ep, t = tab["shard"][i], int(tab["episode"][i]), int(tab["t"][i])
                wm, sim = zarrs[sh]
                w = wm.get_episode(ep)["img"]; s = sim.get_episode(ep)["img"]
                frames = [np.concatenate([w[k], s[k]], axis=0)
                          for k in range(t - 1, min(t + 15, len(w)), 4)]
                strip = np.concatenate(frames, axis=1)
                lab = np.full((18, strip.shape[1], 3), 255, np.uint8)
                cv2.putText(lab, f"{sh}/ep{ep}/t{t} drift{d} psnr{tab['psnr_obs'][i]:.1f} "
                            f"kl{tab['kl'][i]:.1f}", (4, 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
                rows_img.append(np.concatenate([lab, strip], axis=0))
            cv2.imwrite(str(out / "gallery" / f"drift{d:02d}_psnrT{ti}.png"),
                        cv2.cvtColor(np.concatenate(rows_img, axis=0), cv2.COLOR_RGB2BGR))
    print(f"wrote {out}/window_table.npz, review.json, plots/, gallery/")


if __name__ == "__main__":
    main()

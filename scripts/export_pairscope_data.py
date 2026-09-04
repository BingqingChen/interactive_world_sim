"""Export the PairScope viewer payload: all scored windows as compact JSON plus a
stratified selection of episodes encoded as side-by-side WM|sim mp4 data-URIs.

Injects the payload into scripts/pairscope.html (template) at /*__DATA__*/null
and writes the final self-contained HTML.

Usage:
  python scripts/export_pairscope_data.py --shards D,E,F [--n_episodes 36] \
      --out /path/to/pairscope_final.html
"""
import argparse
import base64
import json
import subprocess
import tempfile
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score_chunk_ad import enumerate_windows  # noqa: E402
from build_h1h2_review import load_shard  # noqa: E402

DP_ROOT = "/home/jacobhb/projects/worth_doing/diffusion_policy"
sys.path.insert(0, DP_ROOT)
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


def encode_pair_mp4(w_frames, s_frames, crf):
    """Hstack W|S frames -> H.264 mp4 bytes (10 fps, yuv420p, faststart)."""
    vid = np.concatenate([w_frames, s_frames], axis=2)  # (T,128,256,3)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw.rgb"
        raw.write_bytes(vid.astype(np.uint8).tobytes())
        out = Path(td) / "out.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{vid.shape[2]}x{vid.shape[1]}",
             "-r", "10", "-i", str(raw), "-c:v", "libx264", "-preset", "veryslow",
             "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(out)], check=True)
        return out.read_bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="D,E,F,G")
    ap.add_argument("--pair_prefix", default="datasets/")
    ap.add_argument("--n_episodes", type=int, default=36)
    ap.add_argument("--crf", type=int, default=27)
    ap.add_argument("--media_budget_mb", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pre = Path(args.pair_prefix)
    shards = args.shards.split(",")

    # ---- window table (same fields the viewer plots) ----
    W = {k: [] for k in ("sh", "ep", "t", "drift", "psnr", "psnrp", "kl", "mse",
                         "prog")}
    ep_info = {}  # (shard, ep) -> dict
    R = None
    for si, shard in enumerate(shards):
        meta, man, ad = load_shard(pre, shard)
        R = man["resync_period"]
        starts = np.r_[0, meta["episode_ends"][:-1]]
        ad_map = {(int(e), int(t)): i
                  for i, (e, t) in enumerate(zip(ad["episode"], ad["t"]))} if ad is not None else {}
        for ep, (s, L) in enumerate(zip(starts, meta["ep_len"])):
            psnr = meta["psnr"][s:s + L]
            ang = np.degrees(meta["gt_angle"][s:s + L])
            ep_info[(shard, ep)] = dict(
                len=int(L), seed=int(meta["env_seed"][ep]),
                mean_psnr=float(np.nanmean(psnr)),
                psnr=np.round(np.nan_to_num(psnr, nan=-1), 1).tolist(),
                angle=np.round(np.nan_to_num(ang, nan=0), 1).tolist(),
                widx=[])
            for t in enumerate_windows(int(L), R):
                i = ad_map.get((ep, t))
                ep_info[(shard, ep)]["widx"].append(len(W["t"]))
                W["sh"].append(si)
                W["ep"].append(ep)
                W["t"].append(t)
                W["drift"].append(t % R)
                W["psnr"].append(round(float(np.mean(psnr[t - 1:t + 1])), 2))
                W["psnrp"].append(round(float(np.mean(psnr[t + 1:t + 9])), 2))
                W["kl"].append(round(float(ad["kl"][i]), 3) if i is not None else None)
                W["mse"].append(round(float(ad["mse"][i]) * 1e4, 4) if i is not None else None)
                W["prog"].append(round(float(ang[min(t + 8, L - 1)] - ang[t]), 1))

    n = len(W["t"])
    kl = np.array([v if v is not None else np.nan for v in W["kl"]], float)
    ps = np.array(W["psnr"], float)

    # ---- stratified episode selection: shard x psnr tercile x length tercile ----
    keys = list(ep_info)
    mp = np.array([ep_info[k]["mean_psnr"] for k in keys])
    ln = np.array([ep_info[k]["len"] for k in keys])
    pt = np.digitize(mp, np.percentile(mp, [33, 66]))
    lt = np.digitize(ln, np.percentile(ln, [33, 66]))
    rng = np.random.default_rng(0)
    chosen = []
    per_cell = max(1, args.n_episodes // (len(shards) * 9) + 1)
    for si in range(len(shards)):
        for a in range(3):
            for b in range(3):
                sel = [i for i, k in enumerate(keys)
                       if k[0] == shards[si] and pt[i] == a and lt[i] == b]
                chosen += [keys[i] for i in rng.choice(sel, min(per_cell, len(sel)),
                                                       replace=False)]
    chosen = chosen[:args.n_episodes]

    # ---- encode videos within budget ----
    zarrs = {s: (ReplayBuffer.copy_from_path(
                     str(pre / f"rotate_t_imagined_pair{s}_dp.zarr"), keys=["img"]),
                 ReplayBuffer.copy_from_path(
                     str(pre / f"rotate_t_simreplay_pair{s}_dp.zarr"), keys=["img"]))
             for s in sorted({k[0] for k in chosen})}
    episodes, media = {}, 0
    for shard, ep in chosen:
        wmz, simz = zarrs[shard]
        mp4 = encode_pair_mp4(wmz.get_episode(ep)["img"], simz.get_episode(ep)["img"],
                              args.crf)
        if media + len(mp4) > args.media_budget_mb * 1e6:
            print(f"  media budget reached at {len(episodes)} episodes")
            break
        media += len(mp4)
        info = ep_info[(shard, ep)]
        episodes[f"{shard}/{ep}"] = dict(
            len=info["len"], seed=info["seed"], mean_psnr=round(info["mean_psnr"], 2),
            psnr=info["psnr"], angle=info["angle"], widx=info["widx"],
            video="data:video/mp4;base64," + base64.b64encode(mp4).decode())
        print(f"  {shard}/{ep}: {info['len']}f {len(mp4) / 1e3:.0f}KB "
              f"(total {media / 1e6:.1f}MB)", flush=True)

    fin = np.isfinite(kl)
    relabel = None
    rj = Path("outputs/relabel_mix/relabel_viz.json")
    if rj.exists():
        relabel = json.loads(rj.read_text())
    scaling = None
    sj = Path("outputs/scaling_v2/scaling_viz.json")
    if sj.exists():
        scaling = json.loads(sj.read_text())
    payload = dict(
        generated=time.strftime("%Y-%m-%d %H:%M"),
        shards=shards, R=R, n_windows=n, n_scored=int(fin.sum()),
        psnr_median=round(float(np.median(ps)), 2),
        kl_median=round(float(np.nanmedian(kl)), 2),
        windows=W, episodes=episodes, relabel=relabel, scaling=scaling)

    tpl = (Path(__file__).parent / "pairscope.html").read_text()
    html = tpl.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    Path(args.out).write_text(html)
    print(f"wrote {args.out}: {len(html) / 1e6:.1f}MB "
          f"({n} windows, {len(episodes)} episodes embedded)")


if __name__ == "__main__":
    main()

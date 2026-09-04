"""Build the balanced expert-v2 demo pool.

Merges (a) the in-feasible-mask episodes of rand800 (rotate_t_1k/train eps
200-1000; mask = datasets/feasible_mask_v1.json) with (b) all top-up demos from
datasets/rotate_t_topup/shard*/ whose post-settle init also lands in the mask
(settle drift can push a few out -- those are dropped and counted).
Top-up shard dirs are converted with the existing convert_rotate_t_to_dp_zarr.py
(subprocess) so the img/state/action mapping stays single-sourced.

Outputs:
  datasets/rotate_t_balanced_dp.zarr
  datasets/balanced_pool_manifest.json (per-cell histogram, min over mask, drops)

Usage: python scripts/data_collection/build_balanced_pool.py
"""
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

IWS = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from collect_topup import cell_of, init_xy_cm  # noqa: E402

DP_ROOT = "/home/jacobhb/projects/worth_doing/diffusion_policy"
sys.path.insert(0, DP_ROOT)
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


def main():
    t0 = time.time()
    m = json.loads((IWS / "datasets/feasible_mask_v1.json").read_text())
    edges = np.array(m["grid_cm"]["edges"])
    B = m["grid_cm"]["n"]
    mask = np.array(m["mask"], bool)

    # (a) rand800: keep in-mask episodes (index by post-settle init from the HDF5s)
    files = sorted(glob.glob(str(IWS / "datasets/rotate_t_1k/train/episode_*.hdf5")),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))[200:1000]
    keep_rand, cnt = [], np.zeros((B, B), int)
    for k, f in enumerate(files):
        c = cell_of(*init_xy_cm(f), edges)
        if c is not None and mask[c]:
            keep_rand.append(k)  # rand800 zarr episode index == k
            cnt[c] += 1
    print(f"rand800: keeping {len(keep_rand)}/800 in-mask episodes")

    # (b) top-up shards -> zarrs via the canonical converter
    topup_zarrs, dropped = [], 0
    for shard_dir in sorted((IWS / "datasets/rotate_t_topup").glob("shard*")):
        if not glob.glob(str(shard_dir / "episode_*.hdf5")):
            continue
        out = IWS / f"datasets/_topup_{shard_dir.name}_dp.zarr"
        subprocess.run(
            [sys.executable, str(IWS / "scripts/data_collection/convert_rotate_t_to_dp_zarr.py"),
             "--in_dir", str(shard_dir), "--out", str(out)], check=True, cwd=IWS)
        keep = []
        for k, f in enumerate(sorted(glob.glob(str(shard_dir / "episode_*.hdf5")),
                                     key=lambda p: int(p.split("_")[-1].split(".")[0]))):
            c = cell_of(*init_xy_cm(f), edges)
            if c is not None and mask[c]:
                keep.append(k)
                cnt[c] += 1
            else:
                dropped += 1
        topup_zarrs.append((out, keep))
        print(f"{shard_dir.name}: {len(keep)} kept, converted")

    rb = ReplayBuffer.create_empty_numpy()
    src = ReplayBuffer.copy_from_path(str(IWS / "datasets/rotate_t_rand800_dp.zarr"))
    for k in keep_rand:
        rb.add_episode(src.get_episode(k))
    del src
    for z, keep in topup_zarrs:
        zb = ReplayBuffer.copy_from_path(str(z))
        for k in keep:
            rb.add_episode(zb.get_episode(k))
        del zb
    rb.save_to_path(str(IWS / "datasets/rotate_t_balanced_dp.zarr"), if_exists="replace")

    feas_counts = cnt[mask]
    manifest = dict(
        mask="feasible_mask_v1.json", n_episodes=rb.n_episodes,
        n_steps=int(rb.n_steps), n_rand800=len(keep_rand),
        n_topup=rb.n_episodes - len(keep_rand), dropped_out_of_mask=dropped,
        per_cell_min=int(feas_counts.min()), per_cell_max=int(feas_counts.max()),
        per_cell_median=int(np.median(feas_counts)),
        counts=[[int(v) for v in row] for row in cnt],
        wall_s=int(time.time() - t0))
    (IWS / "datasets/balanced_pool_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != "counts"}, indent=1))


if __name__ == "__main__":
    main()

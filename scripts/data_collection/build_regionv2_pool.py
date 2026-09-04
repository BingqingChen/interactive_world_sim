"""Build the region-v2 real-demo pool for the repeated scaling-law experiment.

Collects every real scripted demo whose POST-SETTLE init lands inside feasible
region v2 (rectangle x in [-2,6] cm, y in [-6,4] cm) from both sources:
  * rand800  (rotate_t_1k/train episodes 200-999)
  * the targeted top-up demos (datasets/rotate_t_topup/shard*)

Episodes are emitted in a fixed, cell-interleaved order so that a contiguous
prefix of length R is approximately uniform over the region's 20 cells -- the
scaling grid slices prefixes (R in {10,20,50,200}), and a naive source order
would make small R cover only a corner of the region.

Output: datasets/rotate_t_regionv2_real_dp.zarr + regionv2_real_manifest.json

Usage: python scripts/data_collection/build_regionv2_pool.py
"""
import glob
import json
import subprocess
import sys
import time
from collections import defaultdict
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
    m = json.loads((IWS / "datasets/feasible_mask_v2.json").read_text())
    edges = np.array(m["grid_cm"]["edges"])
    mask = np.array(m["mask"], bool)

    # ---- gather (source, index, cell) for every in-region demo ----
    entries = []
    r800 = sorted(glob.glob(str(IWS / "datasets/rotate_t_1k/train/episode_*.hdf5")),
                  key=lambda p: int(p.split("_")[-1].split(".")[0]))[200:1000]
    for k, f in enumerate(r800):
        c = cell_of(*init_xy_cm(f), edges)
        if c is not None and mask[c]:
            entries.append(("rand800", k, c))
    topup_specs = []
    for shard_dir in sorted((IWS / "datasets/rotate_t_topup").glob("shard*")):
        files = sorted(glob.glob(str(shard_dir / "episode_*.hdf5")),
                       key=lambda p: int(p.split("_")[-1].split(".")[0]))
        if not files:
            continue
        z = IWS / f"datasets/_topup_{shard_dir.name}_dp.zarr"
        if not z.exists():
            subprocess.run(
                [sys.executable, str(IWS / "scripts/data_collection/convert_rotate_t_to_dp_zarr.py"),
                 "--in_dir", str(shard_dir), "--out", str(z)], check=True, cwd=IWS)
        topup_specs.append((shard_dir.name, z))
        for k, f in enumerate(files):
            c = cell_of(*init_xy_cm(f), edges)
            if c is not None and mask[c]:
                entries.append((shard_dir.name, k, c))
    print(f"{len(entries)} in-region demos "
          f"({sum(1 for e in entries if e[0] == 'rand800')} rand800, "
          f"{sum(1 for e in entries if e[0] != 'rand800')} top-up)")

    # ---- cell-interleaved ordering: round-robin over cells ----
    by_cell = defaultdict(list)
    for e in entries:
        by_cell[e[2]].append(e)
    rng = np.random.default_rng(0)
    for c in by_cell:
        rng.shuffle(by_cell[c])
    cells = sorted(by_cell)
    ordered = []
    while any(by_cell[c] for c in cells):
        for c in cells:
            if by_cell[c]:
                ordered.append(by_cell[c].pop())
    print(f"ordered {len(ordered)} episodes round-robin over {len(cells)} cells "
          f"(prefix R=20 covers {len(set(e[2] for e in ordered[:20]))} cells)")

    # ---- emit ----
    srcs = {"rand800": ReplayBuffer.copy_from_path(
        str(IWS / "datasets/rotate_t_rand800_dp.zarr"))}
    for name, z in topup_specs:
        srcs[name] = ReplayBuffer.copy_from_path(str(z))
    rb = ReplayBuffer.create_empty_numpy()
    for src, idx, _ in ordered:
        ep = srcs[src].get_episode(idx)
        rb.add_episode({k: ep[k] for k in ("img", "state", "action")})
    outz = IWS / "datasets/rotate_t_regionv2_real_dp.zarr"
    rb.save_to_path(str(outz), if_exists="replace")

    hist = defaultdict(int)
    for _, _, c in ordered:
        hist[f"{c[0]},{c[1]}"] += 1
    manifest = dict(
        region="feasible_mask_v2.json (x in [-2,6], y in [-6,4] cm)",
        n_episodes=rb.n_episodes, n_steps=int(rb.n_steps),
        n_rand800=sum(1 for e in ordered if e[0] == "rand800"),
        n_topup=sum(1 for e in ordered if e[0] != "rand800"),
        ordering="round-robin over cells (prefix of length R is ~uniform over the region)",
        per_cell_counts=dict(sorted(hist.items())),
        prefix_cell_coverage={str(R): len(set(e[2] for e in ordered[:R]))
                              for R in (10, 20, 50, 200)},
        wall_s=int(time.time() - t0))
    (IWS / "datasets/regionv2_real_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_cell_counts"}, indent=1))


if __name__ == "__main__":
    main()

"""Targeted top-up collection: bring every feasible cell of the init grid up to
a target demo count using the (unmodified) scripted policy, with the pose
sampler pinned per 2 cm cell.

Reads datasets/feasible_mask_v1.json (the canonical protocol artifact) and the
current per-cell counts of the rand800 pool + any prior top-up output, then for
each feasible cell below --target runs collect_rotate_t.py restricted to that
cell with a bounded trial budget. Demos are assigned to cells by their
POST-SETTLE episode-0 env_state (settle drift moves the T across borders), and
the driver iterates: collect -> recount by assignment -> re-target, up to
--passes rounds. Cells that exhaust their trial cap at <5% accept are flagged
`infeasible_discovered` in the manifest (mask-v2 decision for the user), never
silently retried.

Shardable: --cells "i%N==k" style partition via --shard/--n_shards (disjoint
output dirs + seeds), so 2-3 drivers can run in parallel.

Usage:
  MUJOCO_GL=egl python scripts/data_collection/collect_topup.py \
      --shard 0 --n_shards 2 [--target 25] [--passes 3]
"""
import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

IWS = Path(__file__).resolve().parent.parent.parent


def cell_of(x_cm, y_cm, edges):
    """Grid cell of a post-settle T position, or None if outside the grid.

    Out-of-grid positions are NOT clamped into an edge cell: settle drift can
    push the T past the +-8cm bound, and clamping silently mixed those into the
    edge cells' statistics (both in pool accounting and in eval masking).
    """
    if not (edges[0] <= x_cm < edges[-1] and edges[0] <= y_cm < edges[-1]):
        return None
    return int(np.digitize(x_cm, edges) - 1), int(np.digitize(y_cm, edges) - 1)


def init_xy_cm(h5path):
    with h5py.File(h5path, "r") as h:
        s = h["env_state"][0]
        return float(s[0]) * 100, float(s[1]) * 100


def counts_now(edges, B, rand800_dir, topup_dirs):
    cnt = np.zeros((B, B), int)
    files = sorted(glob.glob(str(rand800_dir / "episode_*.hdf5")),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))[200:]
    for f in files:
        c = cell_of(*init_xy_cm(f), edges)
        if c is None:
            continue  # settled outside the grid -> counts toward no cell
        cnt[c] += 1
    for d in topup_dirs:
        for f in glob.glob(str(d / "episode_*.hdf5")):
            c = cell_of(*init_xy_cm(f), edges)
            if c is None:
                continue
            cnt[c] += 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", default=str(IWS / "datasets/feasible_mask_v1.json"))
    ap.add_argument("--target", type=int, default=25)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--trial_cap_mult", type=int, default=20)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--rand800_dir", default=str(IWS / "datasets/rotate_t_1k/train"))
    ap.add_argument("--out_root", default=str(IWS / "datasets/rotate_t_topup"))
    args = ap.parse_args()

    m = json.loads(Path(args.mask).read_text())
    edges = np.array(m["grid_cm"]["edges"])
    B = m["grid_cm"]["n"]
    mask = np.array(m["mask"], bool)
    out_dir = Path(args.out_root) / f"shard{args.shard}"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_topup_dirs = sorted(Path(args.out_root).glob("shard*"))

    stats = {}  # (i,j) -> dict(attempts, successes)
    t0 = time.time()
    for p in range(args.passes):
        cnt = counts_now(edges, B, Path(args.rand800_dir), all_topup_dirs)
        todo = [(i, j, args.target - cnt[i, j]) for i in range(B) for j in range(B)
                if mask[i, j] and cnt[i, j] < args.target
                and (i * B + j) % args.n_shards == args.shard
                and not stats.get((i, j), {}).get("capped")]
        if not todo:
            print(f"pass {p}: nothing to do for shard {args.shard}")
            break
        print(f"pass {p}: {len(todo)} cells, total deficit "
              f"{sum(t[2] for t in todo)} ({time.time()-t0:.0f}s)", flush=True)
        for i, j, need in todo:
            xlo, xhi = edges[i] / 100, edges[i + 1] / 100
            ylo, yhi = edges[j] / 100, edges[j + 1] / 100
            cap = args.trial_cap_mult * need
            # collector computes ep_seed = seed*1e6 + trial, which must stay
            # < 2**32 -> CLI seed must stay < 4294. Unique per (cell, pass, shard).
            seed = 1000 + (i * B + j) + 64 * p + 200 * args.shard
            before = len(glob.glob(str(out_dir / "episode_*.hdf5")))
            print(f"  cell ({edges[i]:.0f},{edges[j]:.0f})cm need {need} "
                  f"cap {cap} seed {seed}", flush=True)
            r = subprocess.run(
                [sys.executable, str(IWS / "scripts/data_collection/collect_rotate_t.py"),
                 "-o", str(out_dir), "-n", str(need), "--headless",
                 "--seed", str(seed), "--max_trials", str(cap),
                 "--x_min", f"{xlo:.4f}", "--x_max", f"{xhi:.4f}",
                 "--y_min", f"{ylo:.4f}", "--y_max", f"{yhi:.4f}"],
                capture_output=True, text=True, cwd=IWS)
            got = len(glob.glob(str(out_dir / "episode_*.hdf5"))) - before
            trials = r.stdout.count("trial") and max(
                [int(t) for t in __import__("re").findall(r"trial (\d+)", r.stdout)] or [0])
            s = stats.setdefault((i, j), {"attempts": 0, "successes": 0})
            s["attempts"] += max(trials, got)
            s["successes"] += got
            rate = s["successes"] / max(s["attempts"], 1)
            if got < need and s["attempts"] >= cap and rate < 0.05:
                s["capped"] = True
                print(f"    INFEASIBLE_DISCOVERED cell ({edges[i]:.0f},{edges[j]:.0f}) "
                      f"accept {rate:.3f}", flush=True)
            print(f"    got {got}/{need} (cum accept {rate:.2f}, "
                  f"{time.time()-t0:.0f}s)", flush=True)

    cnt = counts_now(edges, B, Path(args.rand800_dir), all_topup_dirs)
    manifest = dict(
        shard=args.shard, n_shards=args.n_shards, target=args.target,
        final_counts=[[int(v) for v in row] for row in cnt],
        cells={f"{i},{j}": v for (i, j), v in stats.items()},
        infeasible_discovered=[k for k, v in stats.items() if v.get("capped")],
        deficient_after=[(i, j, int(args.target - cnt[i, j]))
                         for i in range(B) for j in range(B)
                         if mask[i, j] and cnt[i, j] < args.target],
        wall_s=int(time.time() - t0))
    (out_dir / "topup_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(json.dumps({k: v for k, v in manifest.items() if k != "final_counts"},
                     indent=1, default=str))


if __name__ == "__main__":
    main()

"""Build the expert-v3 training pool: region-v3 real demos + filtered self-imitation.

Region v3 is the full [-6,6]^2 init box the stronger expert must cover. The scripted
planner cannot supply demos across all of it -- it scores ~3/19 in the x<-2 band, so
the existing real pool is severely depleted there (the x=-6 column has 0-2 demos per
cell). Filtered self-imitation fills that hole: the LEARNED expert succeeds ~8/12 in
the same band, and keeping only its successful rollouts is a binary advantage filter
that needs no action log-probs (so it works with a diffusion policy, where PPO does
not).

Two sources, merged cell-interleaved so the pool is as close to uniform as the data
allows:
  * real   -- ALL randomised-init scripted demos (rand800 + top-up), NOT filtered to
              region v3: out-of-region successes still widen the state distribution
              the policy is robust to. Eval still targets region v3.
              (--region_only restores the old in-mask-only behaviour.)
  * selfim -- successful policy rollouts from datasets/selfimit/r<round>/*.zarr

--cap_per_cell trims over-represented cells (the +x half) so the depleted band is not
drowned out; --max_selfim_frac bounds how much of the pool is model-generated.

Output: datasets/rotate_t_expertv3_pool.zarr + expertv3_pool_manifest.json

Usage: python scripts/data_collection/build_expertv3_pool.py --rounds 1 --cap_per_cell 40
"""
import argparse
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


OUT_OF_GRID = (-1, -1)  # pseudo-cell for demos outside the 8x8 grid entirely


def real_entries(edges, mask, region_only=False):
    """(source_key, episode_idx, cell) for every randomised-init scripted demo.

    By default NO region filter is applied: demos outside region v3 are kept because
    they are still successful expert trajectories and widen the state distribution
    the policy is robust to. Evaluation still targets region v3; only the training
    pool is broadened. Pass region_only=True to restore the old in-mask-only pool.

    Episodes 0-199 of rotate_t_1k are always excluded: they are the FIXED-init set,
    all 200 sitting at exactly (0,0). Including them would pile 200 identical inits
    into a single cell and skew the pool badly -- this is why the historical slice
    starts at 200, and it is not an oversight to be "fixed".
    """
    out, specs = [], []
    r800 = sorted(glob.glob(str(IWS / "datasets/rotate_t_1k/train/episode_*.hdf5")),
                  key=lambda p: int(p.split("_")[-1].split(".")[0]))[200:1000]

    def keep(f):
        c = cell_of(*init_xy_cm(f), edges)
        if region_only:
            return c if (c is not None and mask[c]) else None
        return c if c is not None else OUT_OF_GRID

    for k, f in enumerate(r800):
        c = keep(f)
        if c is not None:
            out.append(("rand800", k, c))
    for shard in sorted((IWS / "datasets/rotate_t_topup").glob("shard*")):
        files = sorted(glob.glob(str(shard / "episode_*.hdf5")),
                       key=lambda p: int(p.split("_")[-1].split(".")[0]))
        if not files:
            continue
        z = IWS / f"datasets/_topup_{shard.name}_dp.zarr"
        if not z.exists():
            subprocess.run([sys.executable,
                            str(IWS / "scripts/data_collection/convert_rotate_t_to_dp_zarr.py"),
                            "--in_dir", str(shard), "--out", str(z)], check=True, cwd=IWS)
        specs.append((shard.name, z))
        for k, f in enumerate(files):
            c = keep(f)
            if c is not None:
                out.append((shard.name, k, c))
    return out, specs


def selfim_entries(rounds, edges, mask):
    """(source_key, episode_idx, cell) for every kept self-imitation rollout.

    Round-1 manifests predate per-episode metadata; those fall back to the worker's
    aggregate cell counts, which is enough for interleaving but not exact per-episode
    placement -- flagged in the manifest as `selfim_cells_exact`.
    """
    out, specs, exact = [], [], True
    for r in rounds:
        for z in sorted((IWS / f"datasets/selfimit/r{r}").glob("*.zarr")):
            man = Path(str(z) + ".manifest.json")
            if not man.exists():
                continue
            m = json.loads(man.read_text())
            key = f"si{r}_{z.stem}"
            specs.append((key, z))
            eps = m.get("episodes")
            if eps:
                for k, e in enumerate(eps):
                    out.append((key, k, tuple(e["cell"])))
            else:
                exact = False
                cells = []
                for cs, (s, _a) in m.get("per_cell", {}).items():
                    cells += [tuple(int(v) for v in cs.split(","))] * s
                for k in range(m["n_kept"]):
                    out.append((key, k, cells[k] if k < len(cells) else (0, 0)))
    return out, specs, exact


def interleave(entries, cap, seed=0):
    """Round-robin over cells so any prefix is ~uniform; cap trims dense cells."""
    by = defaultdict(list)
    for e in entries:
        by[e[2]].append(e)
    rng = np.random.default_rng(seed)
    for c in by:
        rng.shuffle(by[c])
        if cap:
            del by[c][cap:]
    cells, out = sorted(by), []
    while any(by[c] for c in cells):
        for c in cells:
            if by[c]:
                out.append(by[c].pop())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=[1])
    ap.add_argument("--region_only", action="store_true",
                    help="restrict real demos to the region-v3 mask (default: keep\n                         all randomised-init demos, wider = more robust)")
    ap.add_argument("--cap_per_cell", type=int, default=40,
                    help="max real demos kept per cell (0 = no cap)")
    ap.add_argument("--selfim_cap_per_cell", type=int, default=0)
    ap.add_argument("--out", default=str(IWS / "datasets/rotate_t_expertv3_pool.zarr"))
    args = ap.parse_args()
    t0 = time.time()

    m = json.loads((IWS / "datasets/feasible_mask_v3.json").read_text())
    edges, mask = np.array(m["grid_cm"]["edges"]), np.array(m["mask"], bool)

    re_, rspecs = real_entries(edges, mask, region_only=args.region_only)
    se_, sspecs, exact = selfim_entries(args.rounds, edges, mask)
    print(f"{len(re_)} real demos ({'region-v3 only' if args.region_only else 'ALL randomised-init, unfiltered'}), "
          f"{len(se_)} self-imitation successes")

    ordered = interleave(re_, args.cap_per_cell) + \
        interleave(se_, args.selfim_cap_per_cell, seed=1)
    ordered = interleave(ordered, 0, seed=2)  # final round-robin mixes both sources

    srcs = {"rand800": ReplayBuffer.copy_from_path(
        str(IWS / "datasets/rotate_t_rand800_dp.zarr"))}
    for k, z in rspecs + sspecs:
        srcs[k] = ReplayBuffer.copy_from_path(str(z))
    rb = ReplayBuffer.create_empty_numpy()
    for src, idx, _ in ordered:
        ep = srcs[src].get_episode(idx)
        rb.add_episode({k: ep[k] for k in ("img", "state", "action")})
    rb.save_to_path(args.out, if_exists="replace")

    hist, shist = defaultdict(int), defaultdict(int)
    for s, _, c in ordered:
        hist[f"{c[0]},{c[1]}"] += 1
        if s.startswith("si"):
            shist[f"{c[0]},{c[1]}"] += 1
    n_si = sum(1 for e in ordered if e[0].startswith("si"))
    man = dict(region=("feasible_mask_v3.json (x,y in [-6,6] cm)" if args.region_only
                       else "UNFILTERED: all randomised-init real demos, any init"),
               rounds=args.rounds, cap_per_cell=args.cap_per_cell,
               n_episodes=rb.n_episodes, n_steps=int(rb.n_steps),
               n_real=len(ordered) - n_si, n_selfim=n_si,
               selfim_frac=round(n_si / max(len(ordered), 1), 3),
               selfim_cells_exact=exact,
               cells_covered=len(hist), cells_total=int(mask.sum()),
               per_cell_counts=dict(sorted(hist.items())),
               per_cell_selfim=dict(sorted(shist.items())),
               wall_s=int(time.time() - t0))
    Path(args.out + ".manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items()
                      if not k.startswith("per_cell")}, indent=1))
    thin = sorted((v, k) for k, v in hist.items())[:6]
    print("thinnest cells:", ", ".join(f"{k}:{v}" for v, k in thin))


if __name__ == "__main__":
    main()

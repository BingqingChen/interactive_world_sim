"""Build a per-cell training zarr for the scaling-v3 sweep (the §4 repeat).

Same slicing contract as build_scaling_zarrs.py -- contiguous episode slices out of
one real pool and one imagined pool, so conditions nest (r10 subset of r20 subset of
r50 ...) and imagined doses are prefixes of a single pool. What changed is where the
pools come from:

  real     datasets/scaling_v3/real/{A..H}.zarr   -- 800 REAL-SIM rollouts of the
           DP+residual expert (96.0% on region v3), success-filtered, inits uniform
           over [-6,6]^2. The published sweep used SCRIPTED demos, whose planner
           fails ~84% of the time at x<-2 and therefore left that band nearly empty;
           collecting with the expert removes that confound.
  imagined datasets/scaling_v3/imag/half{1,2}.zarr -- 800 world-model rollouts of the
           SAME expert, region-v3 inits, success-filtered by the terminal-frame
           detector (the imagined arm's filter is a proxy read off hallucinated
           frames, not ground truth -- see the experiment log).

Shards are concatenated in sorted filename order into one flat index, so a slice is
reproducible from the manifest alone.

Usage:
  python scripts/data_collection/build_scaling_v3_zarrs.py \
      --real_slice 0:10 --imag_slice 0:400 --out datasets/tmp_v3_r10i400.zarr
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

IWS = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, "/home/jacobhb/projects/worth_doing/diffusion_policy")
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


def parse_slice(s):
    a, b = s.split(":")
    return int(a), int(b)


def shard_list(pattern):
    """Sorted shards + cumulative episode offsets, so index k is well defined."""
    out, total = [], 0
    for z in sorted(glob.glob(pattern)):
        rb = ReplayBuffer.copy_from_path(z)
        out.append((z, rb, total, total + rb.n_episodes))
        total += rb.n_episodes
    return out, total


def take(shards, lo, hi, rb_out):
    """Append pooled episodes [lo, hi) onto rb_out."""
    n = 0
    for _z, rb, s0, s1 in shards:
        if hi <= s0 or lo >= s1:
            continue
        for i in range(max(lo, s0), min(hi, s1)):
            ep = rb.get_episode(i - s0)
            rb_out.add_episode({k: ep[k] for k in ("img", "state", "action")})
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_glob", default=str(IWS / "datasets/scaling_v3/real/*.zarr"))
    ap.add_argument("--imag_glob", default=str(IWS / "datasets/scaling_v3/imag/*.zarr"))
    ap.add_argument("--real_slice", required=True)
    ap.add_argument("--imag_slice", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()

    real, n_real = shard_list(args.real_glob)
    imag, n_imag = shard_list(args.imag_glob)
    ra, rb_hi = parse_slice(args.real_slice)
    ia, ib = parse_slice(args.imag_slice)
    for name, hi, avail in (("real", rb_hi, n_real), ("imagined", ib, n_imag)):
        if hi > avail:
            raise SystemExit(f"{name} slice needs {hi} episodes but pool has {avail}")

    out = ReplayBuffer.create_empty_numpy()
    nr = take(real, ra, rb_hi, out)
    ni = take(imag, ia, ib, out)
    out.save_to_path(args.out, if_exists="replace")
    man = dict(real_slice=args.real_slice, imag_slice=args.imag_slice,
               n_real=nr, n_imagined=ni, n_episodes=out.n_episodes,
               n_steps=int(out.n_steps), real_pool=n_real, imag_pool=n_imag,
               real_shards=[Path(z).name for z, *_ in real],
               imag_shards=[Path(z).name for z, *_ in imag],
               wall_s=int(time.time() - t0))
    Path(args.out + ".manifest.json").write_text(json.dumps(man, indent=1))
    print(f"{out.n_episodes} episodes ({nr} real + {ni} imagined), "
          f"{out.n_steps} steps -> {args.out}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

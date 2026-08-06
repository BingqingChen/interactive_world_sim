"""Build a per-run training zarr for the scaling sweep from pooled source zarrs.

Sources (machine-agnostic — pool zarrs, not HDF5):
  real pool:      rotate_t_rand800_dp.zarr   (episode i = rotate_t_1k/train episode 200+i)
  imagined pools: rotate_t_imagined_randA_dp.zarr (375) + randB (375) + randC (50)
                  concatenated in that order -> pooled imagined indices 0..799

Usage:
  python scripts/data_collection/build_scaling_zarrs.py \
      --real_slice 0:10 --imag_slice 0:400 --out datasets/tmp_r10_i400.zarr
  (either slice may be empty: "0:0")
"""
import argparse
import sys
from pathlib import Path

DP_ROOT_CANDIDATES = [
    "/home/jacobhb/projects/worth_doing/diffusion_policy",
    "/home/jacob/projects/worth_doing/diffusion_policy",
]
for p in DP_ROOT_CANDIDATES:
    if Path(p).exists():
        sys.path.insert(0, p)
        break
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


def parse_slice(s):
    a, b = s.split(":")
    return int(a), int(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_dir", default="datasets")
    ap.add_argument("--real_slice", required=True, help="a:b into the 800-real pool")
    ap.add_argument("--imag_slice", required=True,
                    help="a:b into the pooled 800 imagined episodes (randA+randB+randC)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = Path(args.datasets_dir)
    rb = ReplayBuffer.create_empty_numpy()

    ra, rb_hi = parse_slice(args.real_slice)
    if rb_hi > ra:
        pool = ReplayBuffer.copy_from_path(str(d / "rotate_t_rand800_dp.zarr"))
        assert rb_hi <= pool.n_episodes, f"real pool has {pool.n_episodes}"
        for i in range(ra, rb_hi):
            rb.add_episode(pool.get_episode(i))
    n_real = rb.n_episodes

    ia, ib = parse_slice(args.imag_slice)
    if ib > ia:
        # Lazy-load pools: only open a pool if the requested range reaches it.
        loaded = {}

        def pool(idx):
            if idx not in loaded:
                name = f"rotate_t_imagined_rand{'ABC'[idx]}_dp.zarr"
                loaded[idx] = ReplayBuffer.copy_from_path(str(d / name))
            return loaded[idx]

        base = 0
        gi = ia
        for pi in range(3):
            if gi >= ib:
                break
            # peek size without keeping unneeded pools: must open to know size,
            # but only pools whose start offset is below ib are ever touched.
            if base >= ib:
                break
            p = pool(pi)
            sz = p.n_episodes
            while gi < ib and gi < base + sz:
                rb.add_episode(p.get_episode(gi - base))
                gi += 1
            base += sz
        assert gi == ib, f"imagined pools exhausted at {gi} (wanted {ib})"

    rb.save_to_path(args.out, if_exists="replace")
    print(f"Wrote {rb.n_episodes} episodes ({n_real} real + "
          f"{rb.n_episodes - n_real} imagined) / {rb.n_steps} steps to {args.out}")


if __name__ == "__main__":
    main()

"""Build one training zarr per (R, I) cell of the region-v2 scaling grid.

Real pool:     datasets/rotate_t_regionv2_real_dp.zarr (591 eps, cell-interleaved,
               so a prefix of length R is ~uniform over the 20 region cells)
Imagined pool: datasets/rotate_t_imagined_v2A.zarr + v2B.zarr concatenated
               (expert-v2 rolled out in the WM from region-v2 inits, its own
               actions as labels, failures kept -- the OFF-POLICY arm)

Both are sliced as contiguous prefixes so conditions nest (R=10 subset of R=20 ...).

Usage:
  python scripts/data_collection/build_scaling_v2_zarrs.py \
      --real 20 --imag 400 --out datasets/tmp_v2_r20i400.zarr
  # or build the whole grid:
  python scripts/data_collection/build_scaling_v2_zarrs.py --grid
"""
import argparse
import json
import sys
from pathlib import Path

IWS = Path(__file__).resolve().parent.parent.parent
DP_ROOT = "/home/jacobhb/projects/worth_doing/diffusion_policy"
sys.path.insert(0, DP_ROOT)
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402

REAL = IWS / "datasets/rotate_t_regionv2_real_dp.zarr"
IMAG = [IWS / "datasets/rotate_t_imagined_v2A.zarr",
        IWS / "datasets/rotate_t_imagined_v2B.zarr"]
R_VALUES = (10, 20, 50, 200)
I_VALUES = (0, 100, 200, 400, 800)


def build(n_real, n_imag, out, real_rb=None, imag_rbs=None):
    rb = ReplayBuffer.create_empty_numpy()
    src = real_rb if real_rb is not None else ReplayBuffer.copy_from_path(str(REAL))
    assert n_real <= src.n_episodes, f"real pool has {src.n_episodes}, need {n_real}"
    for i in range(n_real):
        rb.add_episode(src.get_episode(i))
    if n_imag:
        pools = imag_rbs if imag_rbs is not None else [
            ReplayBuffer.copy_from_path(str(p)) for p in IMAG]
        taken = 0
        for p in pools:
            for i in range(p.n_episodes):
                if taken >= n_imag:
                    break
                rb.add_episode(p.get_episode(i))
                taken += 1
            if taken >= n_imag:
                break
        assert taken == n_imag, f"imagined pools exhausted at {taken}, wanted {n_imag}"
    rb.save_to_path(str(out), if_exists="replace")
    return rb.n_episodes, int(rb.n_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=int)
    ap.add_argument("--imag", type=int)
    ap.add_argument("--out")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--out_dir", default=str(IWS / "datasets"))
    args = ap.parse_args()

    if args.grid:
        real_rb = ReplayBuffer.copy_from_path(str(REAL))
        imag_rbs = [ReplayBuffer.copy_from_path(str(p)) for p in IMAG]
        print(f"real pool {real_rb.n_episodes} eps; imagined pools "
              f"{[p.n_episodes for p in imag_rbs]} eps")
        manifest = {}
        for R in R_VALUES:
            for I in I_VALUES:
                out = Path(args.out_dir) / f"tmp_v2_r{R}i{I}.zarr"
                n_ep, n_st = build(R, I, out, real_rb, imag_rbs)
                manifest[f"r{R}i{I}"] = dict(episodes=n_ep, steps=n_st, path=str(out))
                print(f"  r{R}i{I}: {n_ep} eps, {n_st} steps")
        (Path(args.out_dir) / "scaling_v2_manifest.json").write_text(
            json.dumps(manifest, indent=1))
    else:
        n_ep, n_st = build(args.real, args.imag, args.out)
        print(f"{args.out}: {n_ep} eps, {n_st} steps")


if __name__ == "__main__":
    main()

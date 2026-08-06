"""Build the imagined+expert mixed DP zarr: all imagined episodes plus the first
N scripted expert episodes (default 20 = 10% of 200) from the fixed-init demos.

Usage:
  python scripts/data_collection/build_imagined_mix_zarr.py \
      --imagined_zarr datasets/rotate_t_imagined_dp.zarr \
      --expert_dir datasets/rotate_t_fixed --n_expert 20 \
      --out datasets/rotate_t_imagined_mix_dp.zarr
"""
import argparse
import sys

import h5py
import numpy as np

DP_ROOT = "/home/jacobhb/projects/worth_doing/diffusion_policy"
sys.path.insert(0, DP_ROOT)
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagined_zarr", default="datasets/rotate_t_imagined_dp.zarr")
    ap.add_argument("--expert_dir", default="datasets/rotate_t_fixed")
    ap.add_argument("--n_expert", type=int, default=20)
    ap.add_argument("--out", default="datasets/rotate_t_imagined_mix_dp.zarr")
    args = ap.parse_args()

    src = ReplayBuffer.copy_from_path(args.imagined_zarr)
    rb = ReplayBuffer.create_empty_numpy()
    for i in range(src.n_episodes):
        rb.add_episode(src.get_episode(i))
    n_imagined = rb.n_episodes

    for ep in range(args.n_expert):
        with h5py.File(f"{args.expert_dir}/episode_{ep}.hdf5", "r") as f:
            img = f["obs/images/top_pov"][:]
            ee = f["obs/ee_pos"][:]
            action = f["action"][:].astype(np.float32)
        state = ee[:, :, :2, 3].reshape(len(ee), -1).astype(np.float32)
        rb.add_episode({"img": img, "state": state, "action": action})

    rb.save_to_path(args.out, if_exists="replace")
    print(f"Wrote {rb.n_episodes} episodes ({n_imagined} imagined + "
          f"{rb.n_episodes - n_imagined} expert) / {rb.n_steps} steps to {args.out}")


if __name__ == "__main__":
    main()

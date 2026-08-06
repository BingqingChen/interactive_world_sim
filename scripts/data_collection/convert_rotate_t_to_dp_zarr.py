"""Convert the rotate-T HDF5 demos into a Diffusion Policy Zarr replay buffer.

Produces a zarr with per-step keys:
  img    (N, 128, 128, 3) uint8   -- top_pov camera (the obs the policy sees)
  state  (N, 4)           float32 -- current bimanual EE xy (proprioception / agent_pos)
  action (N, 4)           float32 -- target bimanual EE xy (what the policy predicts)
plus meta/episode_ends, i.e. the exact format diffusion_policy's image datasets consume.

Usage:
  python scripts/data_collection/convert_rotate_t_to_dp_zarr.py \
      --in_dir datasets/rotate_t --out datasets/rotate_t_dp.zarr
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

# diffusion_policy lives as a sibling checkout.
DP_ROOT = "/home/jacobhb/projects/worth_doing/diffusion_policy"
sys.path.insert(0, DP_ROOT)
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="datasets/rotate_t")
    ap.add_argument("--out", default="datasets/rotate_t_dp.zarr")
    args = ap.parse_args()

    files = sorted(
        glob.glob(os.path.join(args.in_dir, "episode_*.hdf5")),
        key=lambda p: int(p.split("_")[-1].split(".")[0]),
    )
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 in {args.in_dir}")

    rb = ReplayBuffer.create_empty_numpy()
    total = 0
    for p in files:
        with h5py.File(p, "r") as f:
            img = f["obs/images/top_pov"][:]  # (N,128,128,3) uint8
            ee = f["obs/ee_pos"][:]  # (N,2,4,4)
            action = f["action"][:].astype(np.float32)  # (N,4)
        state = ee[:, :, :2, 3].reshape(len(ee), -1).astype(np.float32)  # (N,4) L_xy,R_xy
        assert img.dtype == np.uint8 and img.shape[1:] == (128, 128, 3)
        assert state.shape[1] == 4 and action.shape[1] == 4
        rb.add_episode({"img": img, "state": state, "action": action})
        total += len(img)

    rb.save_to_path(args.out, if_exists="replace")
    print(f"Wrote {rb.n_episodes} episodes / {total} steps to {args.out}")
    print(f"  keys: img{rb['img'].shape} state{rb['state'].shape} action{rb['action'].shape}")
    print(f"  episode lengths: min={np.diff(np.r_[0, rb.episode_ends[:]]).min()} "
          f"max={np.diff(np.r_[0, rb.episode_ends[:]]).max()}")


if __name__ == "__main__":
    main()

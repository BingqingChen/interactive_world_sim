"""Trajectory-replay baseline for the FIXED-init rotate-T task.

Replays the recorded target-EE-xy action sequence of a successful demo through the
same PID/IK controller used everywhere else (closed-loop at the controller level,
open-loop w.r.t. perception -- no policy, no images). One trial = one source demo
replayed against a fresh eval-protocol reset (fixed T at (0,0) upright, stabilize,
no settle, seed 7000+trial).

If this scores high, the fixed-init task is solvable by pure memorization and the
BC numbers must be read in that light; if it scores low, contact/slippage chaos
breaks verbatim replay even from an identical start, and perception is doing real
work.

Usage:
  MUJOCO_GL=egl python scripts/eval_replay_baseline.py \
      --demo_dir datasets/rotate_t_fixed_v2 --n_trials 10 --seed 7000
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "data_collection"))

import eval_dp_rotate_t as E  # noqa: E402
import collect_rotate_t as C  # noqa: E402
import gym_aloha.env as gae  # noqa: E402
from gym_aloha.env import AlohaEnv  # noqa: E402
from yixuan_utilities.kinematics_helper import KinHelper  # noqa: E402
from sim_aloha_dataset_collection_scripted import (  # noqa: E402
    env_state_to_mat,
    trajectory_to_joint_actions,
)
from interactive_world_sim.utils.pose_utils import PoseType, pose_convert  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_dir", default="datasets/rotate_t_fixed_v2")
    ap.add_argument("--n_trials", type=int, default=10,
                    help="Trial k replays demo episode_k against reset seed+k.")
    ap.add_argument("--seed", type=int, default=7000)
    args = ap.parse_args()

    kin = KinHelper(robot_name="trossen_vx300s")
    env = AlohaEnv("pusht")
    gae.sample_pusht_pose = C.make_upright_pose_sampler((0.0, 0.0), (0.0, 0.0))

    succ, deltas = [], []
    for k in range(args.n_trials):
        with h5py.File(f"{args.demo_dir}/episode_{k}.hdf5", "r") as f:
            actions = f["action"][:].astype(np.float64)  # (T,4) recorded targets

        env.reset(seed=args.seed + k)
        C.stabilize_t(env)
        o = env._env.task.get_observation(env._env.physics)
        lb = pose_convert(o["left_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
        rb = pose_convert(o["right_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
        world_t_bases = np.stack([lb, rb])
        curr_vel = np.zeros(6)
        init_pose = env_state_to_mat(
            env._env.task.get_observation(env._env.physics)["env_state"])

        for target_xy in actions:
            obs = env._env.task.get_observation(env._env.physics)
            joint, _ = trajectory_to_joint_actions(
                target_xy, world_t_bases, kin, obs["qpos"][:14],
                curr_vel, E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM)
            env.step(joint)

        final_pose = env_state_to_mat(
            env._env.task.get_observation(env._env.physics)["env_state"])
        ok, d = E.eval_success(init_pose, final_pose)
        succ.append(ok)
        deltas.append(np.degrees(d))
        print(f"  trial {k}: replayed episode_{k} ({len(actions)} steps) -> "
              f"rotated {np.degrees(d):+6.1f} deg  {'SUCCESS' if ok else 'fail'}",
              flush=True)

    sr = float(np.mean(succ))
    deltas = np.array(deltas)
    print("\n========== REPLAY BASELINE ==========")
    print(f"SUCCESS RATE: {sr*100:.0f}%  ({sum(succ)}/{len(succ)})")
    print(f"rotation deg: mean={deltas.mean():+.1f} std={deltas.std():.1f} "
          f"min={deltas.min():+.1f} max={deltas.max():+.1f}")


if __name__ == "__main__":
    main()

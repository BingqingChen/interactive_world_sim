"""Persistent real-env reset server, run under IWS's OWN venv (not RLinf's) --
works around a native dm_control(EGL)+sapien(IK) conflict that segfaults when both
run in RLinf's venv, even with identical package versions to IWS's own (see the plan
doc's "Blocking issue" section for the full bisection).

Pays the heavy AlohaEnv/KinHelper/pose-sampler/feasible-mask construction cost ONCE at
startup, then answers reset requests over stdin/stdout for the rest of its life --
critical for real RLPD training, where batch-synchronous reset (see
world_model_iws_rotate_t_env.py) means every row's episode end triggers a whole-batch
reset, so a full training run needs on the order of 1000+ real resets. A fresh
subprocess per reset (spawn + heavy imports every time) would dwarf the WM's own
diffusion-sampling cost; this amortizes that cost to one-time startup.

Protocol (binary, over stdin/stdout pipes):
  request:  client writes exactly 1 byte (any value) to signal "give me one reset"
  response: server writes an 8-byte big-endian length header, then that many bytes of
            pickle.dumps((image, home_xy, frame_u8)) -- image f32 (3,128,128) [0,1],
            home_xy f32 (4,), frame_u8 (128,128,3) uint8. Matches
            IWSRotateTWorldEnv._one_real_reset's return contract exactly.
  EOF on stdin (client closes it) -> server exits cleanly.

Usage (spawned by IWSRotateTWorldEnv._build_dataset, not run manually):
  MUJOCO_GL=egl <iws_venv>/bin/python real_reset_server.py \
      --x_min -0.06 --x_max 0.06 --y_min -0.06 --y_max 0.06 \
      --feasible_mask <path> --env_seed_base <int>
"""
import argparse
import json
import os
import pickle
import struct
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_min", type=float, default=-0.06)
    ap.add_argument("--x_max", type=float, default=0.06)
    ap.add_argument("--y_min", type=float, default=-0.06)
    ap.add_argument("--y_max", type=float, default=0.06)
    ap.add_argument("--feasible_mask", default=None)
    ap.add_argument("--env_seed_base", type=int, default=700_000)
    ap.add_argument("--iws_scripts_root", required=True,
                    help="the worktree/checkout root whose scripts/ this server imports from")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    root = Path(args.iws_scripts_root)
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(root / "scripts" / "data_collection"))

    import eval_dp_rotate_t as E
    import collect_rotate_t as C
    import gym_aloha.env as gae
    from gym_aloha.env import AlohaEnv
    from yixuan_utilities.kinematics_helper import KinHelper
    from interactive_world_sim.utils.pose_utils import PoseType, pose_convert

    gae.sample_pusht_pose = C.make_upright_pose_sampler(
        (args.x_min, args.x_max), (args.y_min, args.y_max))
    env = AlohaEnv("pusht")
    kin = KinHelper(robot_name="trossen_vx300s")

    mask = edges = None
    if args.feasible_mask:
        m = json.loads(Path(args.feasible_mask).read_text())
        mask = np.array(m["mask"], bool)
        edges = np.array(m["grid_cm"]["edges"])

    def in_feasible_region():
        if mask is None:
            return True
        s = env._env.task.get_observation(env._env.physics)["env_state"]
        x_cm, y_cm = s[0] * 100, s[1] * 100
        if not (edges[0] <= x_cm < edges[-1] and edges[0] <= y_cm < edges[-1]):
            return False
        i, j = int(np.digitize(x_cm, edges) - 1), int(np.digitize(y_cm, edges) - 1)
        return bool(mask[i, j])

    trial = 0

    def one_reset():
        nonlocal trial
        while True:
            np.random.seed(args.env_seed_base + trial)
            env.reset(seed=args.env_seed_base + trial)
            trial += 1
            C.stabilize_t(env)
            o = env._env.task.get_observation(env._env.physics)
            lb = pose_convert(o["left_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            rb = pose_convert(o["right_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            world_t_bases = np.stack([lb, rb])
            C.settle_arms(env, world_t_bases, kin, np.zeros(6),
                          E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM)
            if abs(C.t_angle(env)) > C.UPRIGHT_TOL:
                continue
            if not in_feasible_region():
                continue
            image, agent_pos, frame_u8 = E.get_obs(env, kin, world_t_bases)
            return image, agent_pos.astype(np.float32), frame_u8

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    sys.stderr.write(f"real_reset_server: ready (pid={os.getpid()})\n")
    sys.stderr.flush()

    while True:
        req = stdin.read(1)
        if not req:
            break  # client closed stdin -> exit
        image, home_xy, frame_u8 = one_reset()
        payload = pickle.dumps((image, home_xy, frame_u8), protocol=4)
        stdout.write(struct.pack(">Q", len(payload)))
        stdout.write(payload)
        stdout.flush()

    sys.stderr.write("real_reset_server: stdin closed, exiting\n")


if __name__ == "__main__":
    main()

"""Persistent real-simulator chunk-step server, run under IWS's OWN venv (not
RLinf's) -- same reason as real_reset_server.py: dm_control(EGL) + SAPIEN's IK
solver segfault when run together in RLinf's venv (root cause not found, see the
plan doc's "Blocking issue" section). Unlike real_reset_server.py (which only needs
to bridge the one real reset per episode), THIS server needs to bridge EVERY chunk
step, since real-sim validation runs actual physics for the whole rollout, not just
frame 0 -- so the entire environment lives in this subprocess, and the RLinf-venv
side (IWSRotateTSimEnv) is a pure IPC client with no MuJoCo/dm_control/SAPIEN
dependency at all, avoiding the segfault entirely rather than working around it.

Reuses AlohaChunkEnv verbatim from ppo_residual_rotate_t.py -- the already-proven
real-sim reward (full eval_success angle+flatness gate, not the WM env's simplified
angle-only approximation) -- rather than re-deriving it and risking the same kind of
subtle discrepancy that turned up between the WM env and this precedent (see the
"remind me the reward function" discussion in the plan doc).

Protocol (binary, over stdin/stdout pipes):
  request:  client writes 1 byte: 0x01 = reset_all, 0x02 = step_chunk
    step_chunk additionally sends an 8-byte big-endian length header + that many
    pickled bytes of a (B, n_act, 4) float32 GENUINE waypoint-chunk array (n_act
    distinct EE-xy targets per env, matching AlohaChunkEnv.step_chunk's own
    n_action_steps=8 convention -- not a single repeated action; see
    IWSRotateTSimEnv.chunk_step's docstring for how RLinf's CNNPolicy, which
    only emits one vector per forward call, is made to produce this: setting
    actor.model.action_dim = 4*n_action_steps and reshaping client-side).
  response: 8-byte big-endian length header + that many pickled bytes of:
    reset_all -> (imgs (B,n_obs,3,128,128) f32 [0,1], aps (B,n_obs,4) f32)
    step_chunk -> (imgs, aps, final_imgs, final_aps, rewards (B,) f32,
                   terminations (B,) bool, truncations (B,) bool, successes (B,) bool)
    where imgs/aps are the POST-STEP (and post-auto-reset, per row, for any row
    whose episode just ended -- matching AlohaChunkEnv/ppo_residual_rotate_t.py's
    own per-row independent reset convention -- see below) and final_imgs/final_aps
    are each row's own PRE-reset terminal obs (== imgs/aps for a row that didn't
    finish this call). RLinf's trajectory builder needs a real terminal next_obs
    for the transition that just ended (append_transitions asserts it's not
    None) -- imgs/aps alone can't supply this for a finished row, since they're
    already the NEXT episode's post-reset obs by the time this responds.
  EOF on stdin -> server exits.

Usage (spawned by IWSRotateTSimEnv._build_dataset, not run manually):
  MUJOCO_GL=egl <iws_venv>/bin/python real_sim_chunk_server.py \
      --num_envs 16 --max_steps 800 --n_obs 2 \
      --x_min -0.06 --x_max 0.06 --y_min -0.06 --y_max 0.06 \
      --feasible_mask <path> --env_seed_base <int> \
      --iws_scripts_root <path>
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
    ap.add_argument("--num_envs", type=int, required=True)
    ap.add_argument("--max_steps", type=int, default=800, help="RAW env steps, not chunks (matches AlohaChunkEnv's own units)")
    ap.add_argument("--n_obs", type=int, default=2)
    ap.add_argument("--x_min", type=float, default=-0.06)
    ap.add_argument("--x_max", type=float, default=0.06)
    ap.add_argument("--y_min", type=float, default=-0.06)
    ap.add_argument("--y_max", type=float, default=0.06)
    ap.add_argument("--feasible_mask", required=True)
    ap.add_argument("--env_seed_base", type=int, default=42)
    ap.add_argument("--iws_scripts_root", required=True)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    root = Path(args.iws_scripts_root)
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(root / "scripts" / "data_collection"))
    sys.path.insert(0, str(Path(__file__).parent))  # for aloha_chunk_env.py (vendored, see its docstring)

    import collect_rotate_t as C  # noqa: E402
    import gym_aloha.env as gae  # noqa: E402
    from aloha_chunk_env import AlohaChunkEnv  # noqa: E402

    gae.sample_pusht_pose = C.make_upright_pose_sampler(
        (args.x_min, args.x_max), (args.y_min, args.y_max))

    m = json.loads(Path(args.feasible_mask).read_text())
    mask = np.array(m["mask"], bool)
    edges = np.array(m["grid_cm"]["edges"])

    B = args.num_envs
    envs = [AlohaChunkEnv(args.env_seed_base + 1000 * i, mask, edges, args.max_steps, args.n_obs)
            for i in range(B)]

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    def send(obj):
        payload = pickle.dumps(obj, protocol=4)
        stdout.write(struct.pack(">Q", len(payload)))
        stdout.write(payload)
        stdout.flush()

    def recv_payload():
        header = stdin.read(8)
        if len(header) < 8:
            return None
        (length,) = struct.unpack(">Q", header)
        return pickle.loads(stdin.read(length))

    sys.stderr.write(f"real_sim_chunk_server: ready, {B} envs, pid={os.getpid()}\n")
    sys.stderr.flush()

    while True:
        req = stdin.read(1)
        if not req:
            break
        if req == b"\x01":  # reset_all
            imgs, aps = [], []
            for e in envs:
                img_h, ap_h = e.reset()
                imgs.append(img_h)
                aps.append(ap_h)
            send((np.stack(imgs), np.stack(aps)))
        elif req == b"\x02":  # step_chunk
            actions = recv_payload()  # (B, n_act, 4) float32 -- genuine waypoint chunk per env
            imgs, aps, final_imgs, final_aps, rewards, terminations, truncations, successes = (
                [], [], [], [], [], [], [], [])
            for i, e in enumerate(envs):
                (img_h, ap_h), rew, done, info = e.step_chunk(actions[i])
                ok = bool(info["success"])
                # Pre-reset (this row's own terminal) obs -- RLinf's trajectory
                # builder needs a real next_obs for the transition that just
                # ended, not the NEXT episode's post-reset obs (append_transitions
                # asserts next_obs is not None; a real crash confirmed the client
                # must actually supply this, not just omit "final_observation" as
                # a silently-safe fallback).
                final_imgs.append(img_h)
                final_aps.append(ap_h)
                if done:
                    # per-row independent reset, matching AlohaChunkEnv's own usage
                    # convention in ppo_residual_rotate_t.py (not batch-sync -- real
                    # resets are cheap, no reason to force it).
                    img_h, ap_h = e.reset()
                imgs.append(img_h)
                aps.append(ap_h)
                rewards.append(rew)
                # termination = genuine task success (zero-bootstrap-value ending);
                # truncation = episode ended for any other reason (step budget, or
                # the raw-angle stopping trigger fired but eval_success said no --
                # e.g. tipped over, or overshot past the accept range).
                terminations.append(bool(done and ok))
                truncations.append(bool(done and not ok))
                successes.append(ok)
            send((np.stack(imgs), np.stack(aps), np.stack(final_imgs), np.stack(final_aps),
                 np.array(rewards, np.float32),
                 np.array(terminations, bool), np.array(truncations, bool),
                 np.array(successes, bool)))
        else:
            sys.stderr.write(f"real_sim_chunk_server: unknown request byte {req!r}\n")
            break

    sys.stderr.write("real_sim_chunk_server: stdin closed, exiting\n")


if __name__ == "__main__":
    main()

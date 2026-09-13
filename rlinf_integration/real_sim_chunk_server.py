"""Persistent real-simulator chunk-step server, run under IWS's OWN venv (not
RLinf's): dm_control(EGL) + SAPIEN's IK solver segfault when run together in
RLinf's venv (root cause not found). The whole environment lives here; the
RLinf-side clients (IWSRotateTSimEnv, evaluate_rlpd_checkpoint.py) are pure IPC
clients. Reuses AlohaChunkEnv verbatim for physics and the ground-truth reward.

Action modes (--action_mode):
  absolute: each row's (n_act, 4) actions are EE-xy targets in meters, fed to
    AlohaChunkEnv.step_chunk unchanged (the original design).
  delta: each row's (n_act, 4) actions are normalized per-step deltas in
    [-1, 1]. The server keeps the last commanded target per row (initialised to
    the settled EE-xy at reset) and integrates target_k = clip(target_{k-1} +
    a_k * max_step, target_lo, target_hi). A policy's tanh output is otherwise
    read as meters: the first RLPD run's exploration spanned [-1, 1] m while the
    demos live in a ~0.25 m band, with 0.67 m jumps between consecutive
    waypoints vs 0.003 m in the demos. max_step defaults to 0.04 m (demo p99.9
    per-step |delta| is 0.039-0.051 per dim); the box is the demo action range
    +-3 cm. build_rlpd_offline_rotate_t.py applies the same transform to demos.

Protocol (binary, over stdin/stdout pipes):
  request: 1 byte, 0x01 = reset_all, 0x02 = step_chunk; step_chunk is followed
    by an 8-byte big-endian length + pickled (B, n_act, 4) float32 actions.
  response: 8-byte big-endian length + pickled
    reset_all  -> (imgs (B,n_obs,3,128,128) f32 [0,1], aps (B,n_obs,4) f32)
    step_chunk -> (imgs, aps, final_imgs, final_aps, rewards (B,), terminations (B,),
                   truncations (B,), successes (B,), done_info list[dict|None])
    imgs/aps are post-step and post-auto-reset (per-row independent reset);
    final_imgs/final_aps are each row's own pre-reset obs, which RLinf needs as
    the terminal next_obs. done_info[i] is None unless row i finished this call,
    else {success, delta_deg, init_flat, final_flat, timeout, steps} -- used to
    split failures into timeout / overshoot / undershoot / tipped.
  EOF on stdin -> exit.
"""
import argparse
import json
import os
import pickle
import struct
import sys
from pathlib import Path

import numpy as np

DEMO_TARGET_LO = [-0.261, -0.254, -0.035, -0.223]
DEMO_TARGET_HI = [0.020, 0.216, 0.262, 0.215]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, required=True)
    ap.add_argument("--max_steps", type=int, default=800, help="RAW env steps, not chunks")
    ap.add_argument("--n_obs", type=int, default=2)
    ap.add_argument("--x_min", type=float, default=-0.06)
    ap.add_argument("--x_max", type=float, default=0.06)
    ap.add_argument("--y_min", type=float, default=-0.06)
    ap.add_argument("--y_max", type=float, default=0.06)
    ap.add_argument("--feasible_mask", required=True)
    ap.add_argument("--env_seed_base", type=int, default=42)
    ap.add_argument("--iws_scripts_root", required=True)
    ap.add_argument("--action_mode", choices=["absolute", "delta"], default="absolute")
    ap.add_argument("--max_step", type=float, default=0.04)
    ap.add_argument("--target_lo", type=float, nargs=4, default=DEMO_TARGET_LO)
    ap.add_argument("--target_hi", type=float, nargs=4, default=DEMO_TARGET_HI)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    root = Path(args.iws_scripts_root)
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(root / "scripts" / "data_collection"))
    sys.path.insert(0, str(Path(__file__).parent))

    import collect_rotate_t as C  # noqa: E402
    import eval_dp_rotate_t as E  # noqa: E402
    import gym_aloha.env as gae  # noqa: E402
    from aloha_chunk_env import AlohaChunkEnv  # noqa: E402
    from sim_aloha_dataset_collection_scripted import env_state_to_mat  # noqa: E402

    gae.sample_pusht_pose = C.make_upright_pose_sampler(
        (args.x_min, args.x_max), (args.y_min, args.y_max))

    m = json.loads(Path(args.feasible_mask).read_text())
    mask = np.array(m["mask"], bool)
    edges = np.array(m["grid_cm"]["edges"])

    B = args.num_envs
    envs = [AlohaChunkEnv(args.env_seed_base + 1000 * i, mask, edges, args.max_steps, args.n_obs)
            for i in range(B)]
    lo = np.array(args.target_lo, np.float64)
    hi = np.array(args.target_hi, np.float64)
    last_target = np.zeros((B, 4), np.float64)

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

    def reset_row(i):
        img_h, ap_h = envs[i].reset()
        last_target[i] = ap_h[-1]
        return img_h, ap_h

    def to_targets(i, a):
        if args.action_mode == "absolute":
            return a
        a = np.clip(np.asarray(a, np.float64), -1.0, 1.0) * args.max_step
        out = np.empty_like(a)
        t = last_target[i].copy()
        for k in range(a.shape[0]):
            t = np.clip(t + a[k], lo, hi)
            out[k] = t
        last_target[i] = t
        return out.astype(np.float32)

    def episode_report(e, ok):
        final = env_state_to_mat(e.env._env.task.get_observation(e.env._env.physics)["env_state"])
        _ok, delta = E.eval_success(e.init_pose, final)
        table_n = np.array([0.0, 0.0, 1.0])
        return dict(
            success=bool(ok), delta_deg=float(np.degrees(delta)),
            init_flat=bool(e.init_pose[:3, 2] @ table_n > 0.95),
            final_flat=bool(final[:3, 2] @ table_n > 0.95),
            timeout=bool(e.steps >= e.max_steps), steps=int(e.steps),
        )

    sys.stderr.write(f"real_sim_chunk_server: ready, {B} envs, action_mode={args.action_mode}, "
                     f"pid={os.getpid()}\n")
    sys.stderr.flush()

    while True:
        req = stdin.read(1)
        if not req:
            break
        if req == b"\x01":
            imgs, aps = [], []
            for i in range(B):
                img_h, ap_h = reset_row(i)
                imgs.append(img_h)
                aps.append(ap_h)
            send((np.stack(imgs), np.stack(aps)))
        elif req == b"\x02":
            actions = recv_payload()
            imgs, aps, final_imgs, final_aps = [], [], [], []
            rewards, terminations, truncations, successes, done_info = [], [], [], [], []
            for i, e in enumerate(envs):
                (img_h, ap_h), rew, done, info = e.step_chunk(to_targets(i, actions[i]))
                ok = bool(info["success"])
                final_imgs.append(img_h)
                final_aps.append(ap_h)
                done_info.append(episode_report(e, ok) if done else None)
                if done:
                    img_h, ap_h = reset_row(i)
                imgs.append(img_h)
                aps.append(ap_h)
                rewards.append(rew)
                # termination = task success (zero bootstrap); truncation = any other
                # ending (step budget, or stop trigger fired but eval_success failed).
                terminations.append(bool(done and ok))
                truncations.append(bool(done and not ok))
                successes.append(ok)
            send((np.stack(imgs), np.stack(aps), np.stack(final_imgs), np.stack(final_aps),
                  np.array(rewards, np.float32), np.array(terminations, bool),
                  np.array(truncations, bool), np.array(successes, bool), done_info))
        else:
            sys.stderr.write(f"real_sim_chunk_server: unknown request byte {req!r}\n")
            break

    sys.stderr.write("real_sim_chunk_server: stdin closed, exiting\n")


if __name__ == "__main__":
    main()

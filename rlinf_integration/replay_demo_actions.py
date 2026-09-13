"""Replay expert demo episodes through the RL action path, to check that the RLPD demo buffer
represents actions the way the online env consumes them.

The demo pool (datasets/scaling_v3/real/{A..H}.zarr) was collected by
scripts/collect_policy_rollouts.py, whose reset sequence depends only on --seed and the
feasible mask (a rollout never consumes a trial), so each kept episode's exact initial
state is recovered by re-running that reset loop and matching the manifest's init_xy_cm.
From that state, three replays:
  a   raw demo absolute EE-xy targets, as the expert executed them -- checks the
      reset and physics reproduce the demo (max |EE - demo state| is reported);
  b2  the converter's delta math on the full episode (build_rlpd_offline_rotate_t.py:
      a_k = clip((target_k - prev) / max_step, -1, 1)), integrated from the replay's
      reset EE exactly as real_sim_chunk_server.to_targets does;
  b1  the actions actually stored in the demo buffer, integrated the same way (the
      buffer stops an episode at its proxy success / early stop, so it may cover fewer
      steps than the demo).
Success = eval_dp_rotate_t.eval_success after the executed steps.

Usage (IWS venv):
  MUJOCO_GL=egl python rlinf_integration/replay_demo_actions.py --shard A --n_episodes 25 \
      --buffer_offset 0 --out_json outputs/replay_demo_actions/A.json
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zarr

WT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WT / "rlinf_integration"))
sys.path.insert(0, str(WT / "scripts"))
sys.path.insert(0, str(WT / "scripts" / "data_collection"))

import eval_dp_rotate_t as E  # noqa: E402
import collect_rotate_t as C  # noqa: E402
import gym_aloha.env as gae  # noqa: E402
from gym_aloha.env import AlohaEnv  # noqa: E402
from yixuan_utilities.kinematics_helper import KinHelper  # noqa: E402
from sim_aloha_dataset_collection_scripted import (  # noqa: E402
    env_state_to_mat, trajectory_to_joint_actions,
)
from interactive_world_sim.utils.pose_utils import PoseType, pose_convert  # noqa: E402
from real_sim_chunk_server import DEMO_TARGET_LO, DEMO_TARGET_HI  # noqa: E402

LO, HI = np.array(DEMO_TARGET_LO, np.float64), np.array(DEMO_TARGET_HI, np.float64)


def reset_with_seed(env, kin, s):
    """One reset attempt exactly as collect_policy_rollouts.main does it."""
    np.random.seed(s)
    env.reset(seed=s)
    C.stabilize_t(env)
    o = env._env.task.get_observation(env._env.physics)
    lb = pose_convert(o["left_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
    rb = pose_convert(o["right_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
    wtb = np.stack([lb, rb])
    C.settle_arms(env, wtb, kin, np.zeros(6), E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM)
    return wtb


def valid_init(env, fmask):
    s = env._env.task.get_observation(env._env.physics)["env_state"]
    x, y = s[0] * 100, s[1] * 100
    e = fmask[1]
    inside = e[0] <= x < e[-1] and e[0] <= y < e[-1]
    if inside:
        inside = bool(fmask[0][int(np.digitize(x, e) - 1), int(np.digitize(y, e) - 1)])
    ok = abs(C.t_angle(env)) <= C.UPRIGHT_TOL and inside
    return ok, [round(float(s[0]) * 100, 2), round(float(s[1]) * 100, 2)]


def run(env, kin, wtb, target_fn, n_steps):
    curr_vel = np.zeros(6)
    init_pose = env_state_to_mat(env._env.task.get_observation(env._env.physics)["env_state"])
    _, ap0, _ = E.get_obs(env, kin, wtb)
    aps, targets, reached = [ap0], [], False
    for t in range(n_steps):
        target = target_fn(t, ap0)
        targets.append(target)
        o = env._env.task.get_observation(env._env.physics)
        joint, _ = trajectory_to_joint_actions(
            target.astype(np.float64), wtb, kin, o["qpos"][:14],
            curr_vel, E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM)
        env.step(joint)
        _, ap, _ = E.get_obs(env, kin, wtb)
        aps.append(ap)
        reached |= C.t_angle(env) <= C.TARGET_ANGLE + C.ANGLE_TOL
    final = env_state_to_mat(env._env.task.get_observation(env._env.physics)["env_state"])
    ok, delta = E.eval_success(init_pose, final)
    return dict(success=bool(ok), delta_deg=round(float(np.degrees(delta)), 1),
                reached=bool(reached), n_steps=n_steps), np.stack(aps), np.stack(targets)


def integrator(deltas, max_step):
    """Targets from normalized per-step deltas, as real_sim_chunk_server.to_targets."""
    state = {}

    def fn(t, ap0):
        if t == 0:
            state["target"] = ap0.astype(np.float64).copy()
        state["target"] = np.clip(
            state["target"] + np.clip(deltas[t], -1.0, 1.0) * max_step, LO, HI)
        return state["target"]
    return fn


def converter_deltas(actions_raw, state0, max_step):
    """build_rlpd_offline_rotate_t.py's delta conversion, per step over the full episode."""
    prev = state0.astype(np.float64).copy()
    out = np.empty_like(actions_raw)
    for k in range(len(actions_raw)):
        out[k] = np.clip((actions_raw[k] - prev) / max_step, -1.0, 1.0)
        prev = np.clip(prev + out[k] * max_step, LO, HI)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--n_episodes", type=int, default=25)
    ap.add_argument("--buffer_dir", default=str(WT / "outputs/rlpd_offline_demo_r200_seed1_none_delta"))
    ap.add_argument("--buffer_offset", type=int, required=True,
                    help="demo-buffer trajectory index of this shard's episode 0")
    ap.add_argument("--max_step", type=float, default=0.04)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    shard_zarr = WT / f"datasets/scaling_v3/real/{args.shard}.zarr"
    man = json.loads(Path(str(shard_zarr) + ".manifest.json").read_text())
    m = json.loads((WT / man["feasible_mask"]).read_text())
    fmask = (np.array(m["mask"], bool), np.array(m["grid_cm"]["edges"]))
    z = zarr.open(str(shard_zarr), mode="r")
    ends = z["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])

    env = AlohaEnv("pusht")
    kin = KinHelper(robot_name="trossen_vx300s")
    gae.sample_pusht_pose = C.make_upright_pose_sampler(
        (man["x_min"], man["x_max"]), (man["y_min"], man["y_max"]))

    kept = man["episodes"]
    n_eps = min(args.n_episodes, len(kept))
    results, trial, j = [], 0, 0
    t0 = time.time()
    for k in range(man["n_attempts_done"]):
        if j >= n_eps:
            break
        for _ in range(200):
            seed = man["seed"] + trial
            trial += 1
            wtb = reset_with_seed(env, kin, seed)
            ok_init, init_xy = valid_init(env, fmask)
            if ok_init:
                break
        if init_xy != kept[j]["init_xy_cm"]:
            continue  # a failed (not kept) attempt
        s, e = int(starts[j]), int(ends[j])
        acts = z["data/action"][s:e].astype(np.float64)
        states = z["data/state"][s:e].astype(np.float64)
        n = len(acts)
        rec = dict(shard=args.shard, episode=j, seed=seed, init_xy_cm=init_xy, n_steps=n,
                   demo_delta_deg=kept[j]["delta_deg"])

        reset_with_seed(env, kin, seed)
        ra, aps, _ = run(env, kin, wtb, lambda t, ap0: acts[t], n)
        ra["max_abs_ee_minus_demo_state_m"] = round(float(np.abs(aps[:n] - states).max()), 4)
        rec["a_raw_targets"] = ra

        reset_with_seed(env, kin, seed)
        d2 = converter_deltas(acts, states[0], args.max_step)
        rb2, _, tg2 = run(env, kin, wtb, integrator(d2, args.max_step), n)
        rb2["max_abs_target_minus_demo_target_m"] = round(float(np.abs(tg2 - acts).max()), 4)
        rec["b2_converter_deltas"] = rb2

        files = glob.glob(str(Path(args.buffer_dir) / f"trajectory_{args.buffer_offset + j}_*.pt"))
        if files:
            buf = torch.load(files[0], weights_only=False)["actions"].reshape(-1, 4).numpy()
            nb = min(len(buf), n)
            reset_with_seed(env, kin, seed)
            rb1, _, _ = run(env, kin, wtb, integrator(buf.astype(np.float64), args.max_step), nb)
            rb1["buffer_steps"] = int(len(buf))
            rec["b1_buffer_actions"] = rb1
        results.append(rec)
        j += 1
        print(f"[{args.shard} {j}/{n_eps}] seed {seed} n={n} "
              f"a={ra['success']} b2={rb2['success']} "
              f"b1={rec.get('b1_buffer_actions', {}).get('success')} "
              f"ee_dev={ra['max_abs_ee_minus_demo_state_m']} ({time.time() - t0:.0f}s)", flush=True)

    summary = {}
    for key in ("a_raw_targets", "b2_converter_deltas", "b1_buffer_actions"):
        vals = [r[key]["success"] for r in results if key in r]
        summary[key] = dict(n=len(vals), success=int(sum(vals)),
                            rate=round(sum(vals) / max(len(vals), 1), 3))
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(dict(summary=summary, episodes=results), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

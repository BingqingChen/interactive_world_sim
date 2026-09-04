"""Roll a DP policy out in the REAL MuJoCo sim and save the SUCCESSFUL episodes as a
DP training zarr -- the data engine for advantage-filtered BC / self-imitation.

Why this exists: the scripted planner cannot reliably solve the x<-2 band, so there
are no expert demos there and plain BC cannot learn it. But the learned policy
occasionally succeeds there on its own. Filtering its rollouts by task success is a
binary advantage filter: keep the trajectories that worked, retrain on them, repeat.
No action log-probs are needed, so this works with a diffusion policy where PPO does
not.

Exploration comes from two sources:
  * the diffusion policy's own sampling stochasticity (free, different every rollout)
  * optional Gaussian noise on the commanded EE targets (--action_noise, metres)

Init sampling can be biased toward the region that needs data (--focus_x_min/max),
since uniform sampling over [-6,6]^2 spends most rollouts where the policy is
already good.

Output: <out_zarr> with img/state/action in the demo format, plus a JSON manifest
recording per-cell attempts and successes (the accept-rate map).

Usage:
  MUJOCO_GL=egl python scripts/collect_policy_rollouts.py --ckpt <ckpt> \
      --n_attempts 200 --action_noise 0.004 \
      --x_min -0.06 --x_max 0.06 --y_min -0.06 --y_max 0.06 \
      --out_zarr datasets/selfimit_round1.zarr
"""
import argparse
import collections
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "data_collection"))
import eval_dp_rotate_t as E  # noqa: E402
import collect_rotate_t as C  # noqa: E402
import gym_aloha.env as gae  # noqa: E402
from gym_aloha.env import AlohaEnv  # noqa: E402
from yixuan_utilities.kinematics_helper import KinHelper  # noqa: E402
from sim_aloha_dataset_collection_scripted import (  # noqa: E402
    env_state_to_mat, trajectory_to_joint_actions,
)
from interactive_world_sim.utils.pose_utils import PoseType, pose_convert  # noqa: E402
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


def _write_manifest(args, kept, att, suc, kept_meta, t0, partial=False):
    """Manifest alongside the zarr. `partial` marks a mid-run flush, so a consumer
    can tell an interrupted round from a completed one."""
    man = dict(vars(args), n_kept=kept, n_attempts_done=sum(att.values()),
               accept_rate=round(kept / max(sum(att.values()), 1), 3),
               per_cell={f"{c[0]},{c[1]}": [suc[c], att[c]] for c in att},
               episodes=kept_meta, partial=partial, wall_s=int(time.time() - t0))
    Path(args.out_zarr + ".manifest.json").write_text(json.dumps(man, indent=1))
    return man


@torch.no_grad()
def rollout(env, policy, kin, world_t_bases, n_obs, n_act, max_steps, action_noise, rng):
    """Run one episode, recording the demo-format trajectory. Returns
    (success, delta_deg, dict(img,state,action))."""
    curr_vel = np.zeros(6)
    init_pose = env_state_to_mat(
        env._env.task.get_observation(env._env.physics)["env_state"])
    policy.reset()
    img0, ap0, fr0 = E.get_obs(env, kin, world_t_bases)
    img_hist = collections.deque([img0] * n_obs, maxlen=n_obs)
    ap_hist = collections.deque([ap0] * n_obs, maxlen=n_obs)
    imgs, states, acts = [fr0], [ap0], []
    steps, reached = 0, False
    while steps < max_steps and not reached:
        obs = {"image": torch.from_numpy(np.stack(img_hist)[None]).to(policy.device),
               "agent_pos": torch.from_numpy(np.stack(ap_hist)[None]).to(policy.device)}
        plan = policy.predict_action(obs)["action"][0].cpu().numpy()  # (n_act,4)
        if action_noise > 0:
            plan = plan + rng.normal(0.0, action_noise, plan.shape)
        for target_xy in plan:
            o = env._env.task.get_observation(env._env.physics)
            joint, _ = trajectory_to_joint_actions(
                target_xy.astype(np.float64), world_t_bases, kin, o["qpos"][:14],
                curr_vel, E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM)
            env.step(joint)
            img, ap, fr = E.get_obs(env, kin, world_t_bases)
            img_hist.append(img); ap_hist.append(ap)
            acts.append(target_xy.astype(np.float32))
            imgs.append(fr); states.append(ap)
            steps += 1
            # Match eval_dp_rotate_t.run_episode EXACTLY: on reaching the target angle
            # the remaining actions of the current chunk still execute, and only then
            # does the episode stop. Breaking out mid-chunk instead would stop at the
            # instant of crossing ~83 deg CW and skip the over-rotation that eval would
            # have seen -- a more lenient filter than the metric being optimised, which
            # would admit eval-failures into the training pool.
            if C.t_angle(env) <= C.TARGET_ANGLE + C.ANGLE_TOL:
                reached = True
            if steps >= max_steps:
                break
    final_pose = env_state_to_mat(
        env._env.task.get_observation(env._env.physics)["env_state"])
    ok, delta = E.eval_success(init_pose, final_pose)
    n = len(acts)
    traj = dict(img=np.stack(imgs[:n]), state=np.stack(states[:n]).astype(np.float32),
                action=np.stack(acts))
    return ok, float(np.degrees(delta)), traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="plain DP checkpoint (mutually exclusive with --residual)")
    ap.add_argument("--residual", default=None,
                    help="PPO residual .pt -- rolls out `frozen DP + residual head`, "
                         "the strongest expert (96.0%% on region v3 vs 87%% for the "
                         "best plain DP). See scripts/residual_policy.py.")
    ap.add_argument("--n_attempts", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--action_noise", type=float, default=0.0,
                    help="std (metres) of Gaussian noise added to commanded EE targets")
    ap.add_argument("--x_min", type=float, default=-0.06)
    ap.add_argument("--x_max", type=float, default=0.06)
    ap.add_argument("--y_min", type=float, default=-0.06)
    ap.add_argument("--y_max", type=float, default=0.06)
    ap.add_argument("--feasible_mask", default=None)
    ap.add_argument("--cells", nargs="*", default=None, metavar="I,J",
                    help="restrict inits to these grid cells, round-robin (e.g. 1,4 "
                         "2,3). Overrides --x_*/--y_*; use to pour attempts into the "
                         "cells a previous round measured as weakest.")
    ap.add_argument("--seed", type=int, default=50000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_zarr", required=True)
    ap.add_argument("--min_len", type=int, default=40)
    ap.add_argument("--save_every", type=int, default=50,
                    help="flush the zarr every N kept episodes so a round can be cut "
                         "short (or survive a crash) without losing everything; 0 "
                         "saves only at the end")
    args = ap.parse_args()

    device = torch.device(args.device)
    if bool(args.ckpt) == bool(args.residual):
        raise SystemExit("give exactly one of --ckpt / --residual")
    if args.residual:
        from residual_policy import load_residual_policy
        policy, n_obs, n_act = load_residual_policy(args.residual, device)
        print(f"expert: residual (iter {policy.iter}) over {policy.targs['ckpt']}",
              flush=True)
    else:
        policy, n_obs, n_act = E.load_policy(args.ckpt, device)
    env = AlohaEnv("pusht")
    kin = KinHelper(robot_name="trossen_vx300s")
    gae.sample_pusht_pose = C.make_upright_pose_sampler(
        (args.x_min, args.x_max), (args.y_min, args.y_max))
    fmask = None
    if args.feasible_mask:
        m = json.loads(Path(args.feasible_mask).read_text())
        fmask = (np.array(m["mask"], bool), np.array(m["grid_cm"]["edges"]))

    def cell(env_):
        s = env_._env.task.get_observation(env_._env.physics)["env_state"]
        x, y = s[0] * 100, s[1] * 100
        if fmask is None:
            return (0, 0), True
        e = fmask[1]
        if not (e[0] <= x < e[-1] and e[0] <= y < e[-1]):
            return None, False
        i, j = int(np.digitize(x, e) - 1), int(np.digitize(y, e) - 1)
        return (i, j), bool(fmask[0][i, j])

    rb = ReplayBuffer.create_empty_numpy()
    rng = np.random.default_rng(args.seed)
    att = collections.Counter(); suc = collections.Counter()
    kept_meta = []  # per KEPT episode, aligned with zarr episode index
    t0 = time.time(); trial = 0; kept = 0; last_flush = -1
    targets = [tuple(int(v) for v in c.split(",")) for c in (args.cells or [])]
    if targets:
        e = fmask[1]
        print(f"cell-targeted: {len(targets)} cells, round-robin", flush=True)

    for k in range(args.n_attempts):
        if targets:  # pin the sampler to this attempt's cell
            i, j = targets[k % len(targets)]
            gae.sample_pusht_pose = C.make_upright_pose_sampler(
                (e[i] / 100, e[i + 1] / 100), (e[j] / 100, e[j + 1] / 100))
        for attempt in range(200):  # settle can push an edge-cell init out of the mask
            np.random.seed(args.seed + trial)
            env.reset(seed=args.seed + trial); trial += 1
            C.stabilize_t(env)
            o = env._env.task.get_observation(env._env.physics)
            lb = pose_convert(o["left_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            rb_ = pose_convert(o["right_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            wtb = np.stack([lb, rb_])
            C.settle_arms(env, wtb, kin, np.zeros(6), E.DT, E.K_P, E.K_V,
                          E.ACC_LIM, E.VEL_LIM)
            c, inside = cell(env)
            if abs(C.t_angle(env)) <= C.UPRIGHT_TOL and inside:
                break
        else:
            print(f"  [skip] attempt {k}: no valid init in 200 tries", flush=True)
            continue
        st = env._env.task.get_observation(env._env.physics)["env_state"]
        init_xy = [round(float(st[0]) * 100, 2), round(float(st[1]) * 100, 2)]
        ok, deg, traj = rollout(env, policy, kin, wtb, n_obs, n_act, args.max_steps,
                                args.action_noise, rng)
        att[c] += 1
        if ok and len(traj["action"]) >= args.min_len:
            rb.add_episode(traj); suc[c] += 1; kept += 1
            kept_meta.append(dict(cell=list(c), init_xy_cm=init_xy,
                                  n_steps=len(traj["action"]), delta_deg=round(deg, 1)))
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{args.n_attempts} attempts, {kept} kept "
                  f"({kept/(k+1)*100:.0f}%), {time.time()-t0:.0f}s", flush=True)
        # Flush periodically: without this the zarr appears only after the final
        # attempt, so killing a slow worker throws away every episode it collected.
        if args.save_every and kept and kept % args.save_every == 0 and rb.n_episodes:
            if kept != last_flush:
                rb.save_to_path(args.out_zarr, if_exists="replace")
                _write_manifest(args, kept, att, suc, kept_meta, t0, partial=True)
                last_flush = kept
                print(f"  [flush] {kept} episodes -> {args.out_zarr}", flush=True)
    if rb.n_episodes:
        rb.save_to_path(args.out_zarr, if_exists="replace")
    edges = fmask[1] if fmask is not None else np.array([0])
    _write_manifest(args, kept, att, suc, kept_meta, t0, partial=False)
    print(f"kept {kept}/{args.n_attempts} successful rollouts -> {args.out_zarr}")
    if fmask is not None:
        print("per-cell success/attempts:")
        for c in sorted(att):
            print(f"  ({edges[c[0]]:+.0f},{edges[c[1]]:+.0f}) {suc[c]}/{att[c]}")


if __name__ == "__main__":
    main()

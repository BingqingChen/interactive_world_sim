"""Evaluate a PPO-trained residual policy DETERMINISTICALLY, on the same protocol as
every other number in this study.

Why this exists: the success rate printed during PPO training is measured while the
residual is being SAMPLED, so it includes exploration noise and understates the
policy. The number that can be compared against the 80% / 72% expert baselines is the
mean action with sampling off, evaluated with eval_dp_rotate_t.py's exact init
sampling, step budget and success predicate.

Reports both the residual policy and (with --with_base) the frozen base policy on the
identical init sequence, so the comparison is paired rather than across separate
random draws -- at n=100 an unpaired comparison cannot resolve the few points that
matter here.

Usage:
  MUJOCO_GL=egl python scripts/eval_residual_policy.py \
      --residual outputs/ppo_v2/residual_latest.pt \
      --n_episodes 100 --with_base --metrics_json outputs/ppo_v2/eval_residual.json
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

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
from ppo_residual_rotate_t import FrozenExpert, ResidualAC  # noqa: E402


@torch.no_grad()
def run(env, kin, wtb, expert, ac, delta_max, max_steps, n_obs):
    """One episode. ac=None runs the frozen base policy unchanged."""
    curr_vel = np.zeros(6)
    init_pose = env_state_to_mat(
        env._env.task.get_observation(env._env.physics)["env_state"])
    expert.policy.reset()
    img, ap, _ = E.get_obs(env, kin, wtb)
    ih = collections.deque([img] * n_obs, maxlen=n_obs)
    ah = collections.deque([ap] * n_obs, maxlen=n_obs)
    steps, reached = 0, False
    while steps < max_steps and not reached:
        feat, chunk = expert(np.stack(ih)[None], np.stack(ah)[None])
        if ac is not None:
            mu, _, _ = ac(feat, chunk)                 # MEAN action, no sampling
            chunk = chunk + (torch.tanh(mu) * delta_max).unsqueeze(1)
        plan = chunk[0].cpu().numpy()
        for target_xy in plan:
            o = env._env.task.get_observation(env._env.physics)
            joint, _ = trajectory_to_joint_actions(
                target_xy.astype(np.float64), wtb, kin, o["qpos"][:14],
                curr_vel, E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM)
            env.step(joint)
            i2, a2, _ = E.get_obs(env, kin, wtb)
            ih.append(i2); ah.append(a2)
            steps += 1
            if C.t_angle(env) <= C.TARGET_ANGLE + C.ANGLE_TOL:
                reached = True
            if steps >= max_steps:
                break
    final = env_state_to_mat(
        env._env.task.get_observation(env._env.physics)["env_state"])
    ok, delta = E.eval_success(init_pose, final)
    return bool(ok), float(np.degrees(delta)), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residual", required=True)
    ap.add_argument("--ckpt", default=None, help="override the base ckpt in the residual file")
    ap.add_argument("--n_episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--with_base", action="store_true",
                    help="also run the frozen base policy on the SAME inits (paired)")
    ap.add_argument("--feasible_mask",
                    default=str(Path(__file__).parent.parent / "datasets/feasible_mask_v3.json"))
    ap.add_argument("--no_crn", dest="crn", action="store_false",
                    help="disable common random numbers (default: on -- both policies\n                         draw identical denoising noise, so a small residual effect\n                         is not buried under the diffusion policy's own sampling)")
    ap.add_argument("--metrics_json", required=True)
    args = ap.parse_args()

    blob = torch.load(args.residual, map_location="cpu")
    targs = blob["args"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    expert = FrozenExpert(args.ckpt or targs["ckpt"], device)
    ac = ResidualAC(expert.feat_dim, expert.n_act * 4).to(device)
    ac.load_state_dict(blob["ac"]); ac.eval()
    delta_max = targs["delta_max"]
    print(f"residual from iter {blob['iter']}, delta_max={delta_max}", flush=True)

    m = json.loads(Path(args.feasible_mask).read_text())
    mask, edges = np.array(m["mask"], bool), np.array(m["grid_cm"]["edges"])
    gae.sample_pusht_pose = C.make_upright_pose_sampler(
        (targs["x_min"], targs["x_max"]), (targs["y_min"], targs["y_max"]))
    env = AlohaEnv("pusht"); kin = KinHelper(robot_name="trossen_vx300s")

    def in_region():
        s = env._env.task.get_observation(env._env.physics)["env_state"]
        x, y = s[0] * 100, s[1] * 100
        if not (edges[0] <= x < edges[-1] and edges[0] <= y < edges[-1]):
            return False
        return bool(mask[int(np.digitize(x, edges) - 1), int(np.digitize(y, edges) - 1)])

    recs, trial, t0 = [], 0, time.time()
    for ep in range(args.n_episodes):
        np.random.seed(args.seed + ep)
        for _ in range(200):
            env.reset(seed=args.seed + trial); trial += 1
            C.stabilize_t(env)
            o = env._env.task.get_observation(env._env.physics)
            lb = pose_convert(o["left_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            rb = pose_convert(o["right_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            wtb = np.stack([lb, rb])
            C.settle_arms(env, wtb, kin, np.zeros(6), E.DT, E.K_P, E.K_V,
                          E.ACC_LIM, E.VEL_LIM)
            if abs(C.t_angle(env)) <= C.UPRIGHT_TOL and in_region():
                break
        s = env._env.task.get_observation(env._env.physics)["env_state"]
        init_xy = [round(float(s[0]) * 100, 2), round(float(s[1]) * 100, 2)]
        qpos0 = env._env.physics.data.qpos.copy()
        qvel0 = env._env.physics.data.qvel.copy()

        # Common random numbers: the diffusion policy samples its own action chunks,
        # so two runs from an identical state diverge on denoising noise alone
        # (measured: 200-300 step swings from a sub-millimetre residual). Seeding
        # torch identically before each run makes both draw the same noise, so the
        # residual is the only difference until the trajectories genuinely diverge.
        if args.crn:
            torch.manual_seed(args.seed + ep)
        ok_r, deg_r, st_r = run(env, kin, wtb, expert, ac, delta_max,
                                args.max_steps, expert.n_obs)
        rec = dict(episode=ep, init_xy_cm=init_xy, residual_success=ok_r,
                   residual_deg=deg_r, residual_steps=st_r)
        if args.with_base:
            # restore the identical starting state so the comparison is paired
            with env._env.physics.reset_context():
                env._env.physics.data.qpos[:] = qpos0
                env._env.physics.data.qvel[:] = qvel0
            if args.crn:
                torch.manual_seed(args.seed + ep)
            ok_b, deg_b, st_b = run(env, kin, wtb, expert, None, delta_max,
                                    args.max_steps, expert.n_obs)
            rec.update(base_success=ok_b, base_deg=deg_b, base_steps=st_b)
        recs.append(rec)
        msg = f"  ep {ep:3d}: residual {'OK ' if ok_r else 'fail'} {deg_r:+6.1f}"
        if args.with_base:
            msg += f" | base {'OK ' if rec['base_success'] else 'fail'} {rec['base_deg']:+6.1f}"
        print(msg + f"  ({time.time()-t0:.0f}s)", flush=True)

    n = len(recs)
    sr = sum(r["residual_success"] for r in recs) / n
    summary = dict(residual=args.residual, iter=blob["iter"], n_episodes=n,
                   success_rate=sr, n_success=sum(r["residual_success"] for r in recs),
                   success_se=float(np.sqrt(sr * (1 - sr) / n)))
    if args.with_base:
        b = sum(r["base_success"] for r in recs) / n
        # paired: only discordant episodes carry information about the difference
        win = sum(1 for r in recs if r["residual_success"] and not r["base_success"])
        lose = sum(1 for r in recs if r["base_success"] and not r["residual_success"])
        summary.update(base_success_rate=b, delta=sr - b,
                       residual_wins=win, base_wins=lose, discordant=win + lose)
    Path(args.metrics_json).write_text(
        json.dumps(dict(summary=summary, episodes=recs), indent=1))
    print(f"\nRESIDUAL {sr*100:.1f}% ({summary['n_success']}/{n})", flush=True)
    if args.with_base:
        print(f"BASE     {summary['base_success_rate']*100:.1f}%", flush=True)
        print(f"paired: residual wins {summary['residual_wins']}, "
              f"base wins {summary['base_wins']} "
              f"({summary['discordant']} discordant of {n})", flush=True)


if __name__ == "__main__":
    main()

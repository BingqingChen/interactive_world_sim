"""Does the diffusion policy survive a SHORT denoising chain?

This gates DPPO. DPPO reframes the denoising chain as an MDP and does PPO over its
per-step Gaussian transitions -- which means backpropagating through the chain, so it
is only tractable with ~10 steps, not the 100 DDPM steps this policy is trained and
evaluated with. If accuracy collapses at 10 steps then DPPO would start from a badly
degraded policy and any gain it shows would be against the wrong baseline.

Measures the SAME checkpoint under several samplers on identical episodes with common
random numbers, so the comparison isolates the sampler:
  * DDPM-100  -- the production setting, the number every other result uses
  * DDIM-K    -- candidate fine-tuning settings

Usage:
  MUJOCO_GL=egl python scripts/check_ddim_viability.py --ckpt <ckpt> \
      --n_episodes 25 --steps 100 20 10 5
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
from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # noqa: E402


def set_sampler(policy, kind, steps, orig):
    """Swap the policy's scheduler/step count in place."""
    if kind == "ddpm":
        policy.noise_scheduler = orig
    else:
        cfg = orig.config
        policy.noise_scheduler = DDIMScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_start=cfg.beta_start, beta_end=cfg.beta_end,
            beta_schedule=cfg.beta_schedule, clip_sample=cfg.clip_sample,
            prediction_type=cfg.prediction_type, set_alpha_to_one=False,
            steps_offset=0)
    policy.num_inference_steps = steps


@torch.no_grad()
def episode(env, kin, wtb, policy, n_obs, max_steps):
    curr_vel = np.zeros(6)
    init = env_state_to_mat(env._env.task.get_observation(env._env.physics)["env_state"])
    policy.reset()
    img, ap, _ = E.get_obs(env, kin, wtb)
    ih = collections.deque([img] * n_obs, maxlen=n_obs)
    ah = collections.deque([ap] * n_obs, maxlen=n_obs)
    steps, reached, t0 = 0, False, time.time()
    while steps < max_steps and not reached:
        obs = {"image": torch.from_numpy(np.stack(ih)[None]).to(policy.device),
               "agent_pos": torch.from_numpy(np.stack(ah)[None]).to(policy.device)}
        plan = policy.predict_action(obs)["action"][0].cpu().numpy()
        for target in plan:
            o = env._env.task.get_observation(env._env.physics)
            joint, _ = trajectory_to_joint_actions(
                target.astype(np.float64), wtb, kin, o["qpos"][:14], curr_vel,
                E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM)
            env.step(joint)
            i2, a2, _ = E.get_obs(env, kin, wtb)
            ih.append(i2); ah.append(a2); steps += 1
            if C.t_angle(env) <= C.TARGET_ANGLE + C.ANGLE_TOL:
                reached = True
            if steps >= max_steps:
                break
    fin = env_state_to_mat(env._env.task.get_observation(env._env.physics)["env_state"])
    ok, delta = E.eval_success(init, fin)
    return bool(ok), float(np.degrees(delta)), steps, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_episodes", type=int, default=25)
    ap.add_argument("--steps", type=int, nargs="+", default=[100, 20, 10, 5])
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--feasible_mask",
                    default=str(Path(__file__).parent.parent / "datasets/feasible_mask_v3.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy, n_obs, _ = E.load_policy(args.ckpt, dev)
    orig = policy.noise_scheduler
    m = json.loads(Path(args.feasible_mask).read_text())
    mask, edges = np.array(m["mask"], bool), np.array(m["grid_cm"]["edges"])
    gae.sample_pusht_pose = C.make_upright_pose_sampler((-0.06, 0.06), (-0.06, 0.06))
    env = AlohaEnv("pusht"); kin = KinHelper(robot_name="trossen_vx300s")

    def in_region():
        s = env._env.task.get_observation(env._env.physics)["env_state"]
        x, y = s[0] * 100, s[1] * 100
        if not (edges[0] <= x < edges[-1] and edges[0] <= y < edges[-1]):
            return False
        return bool(mask[int(np.digitize(x, edges) - 1), int(np.digitize(y, edges) - 1)])

    # configs: production DDPM-100 first, then DDIM at each step count
    cfgs = [("ddpm", 100)] + [("ddim", k) for k in args.steps if k != 100]
    res = {f"{k}-{s}": [] for k, s in cfgs}
    trial = 0
    for ep in range(args.n_episodes):
        np.random.seed(args.seed + ep)
        for _ in range(200):
            env.reset(seed=args.seed + trial); trial += 1
            C.stabilize_t(env)
            o = env._env.task.get_observation(env._env.physics)
            wtb = np.stack([pose_convert(o["left_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0],
                            pose_convert(o["right_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]])
            C.settle_arms(env, wtb, kin, np.zeros(6), E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM)
            if abs(C.t_angle(env)) <= C.UPRIGHT_TOL and in_region():
                break
        q0 = env._env.physics.data.qpos.copy(); v0 = env._env.physics.data.qvel.copy()
        line = f"  ep {ep:3d}:"
        for kind, steps in cfgs:
            with env._env.physics.reset_context():
                env._env.physics.data.qpos[:] = q0
                env._env.physics.data.qvel[:] = v0
            set_sampler(policy, kind, steps, orig)
            torch.manual_seed(args.seed + ep)          # CRN across samplers
            ok, deg, st, wall = episode(env, kin, wtb, policy, n_obs, args.max_steps)
            res[f"{kind}-{steps}"].append(dict(ok=ok, deg=deg, steps=st, wall=wall))
            line += f"  {kind}{steps}:{'OK ' if ok else 'fail'}{deg:+6.1f}"
        print(line, flush=True)

    print("\n=== sampler comparison (same episodes, CRN) ===")
    print(f"{'sampler':12s} {'success':>9s} {'mean|rot|':>10s} {'s/episode':>10s}")
    summary = {}
    for k, v in res.items():
        sr = sum(x["ok"] for x in v) / len(v)
        summary[k] = dict(success_rate=sr, n=len(v),
                          mean_wall=float(np.mean([x["wall"] for x in v])))
        print(f"{k:12s} {sr*100:8.1f}% {np.mean([abs(x['deg']) for x in v]):10.1f} "
              f"{np.mean([x['wall'] for x in v]):10.1f}")
    Path(args.out).write_text(json.dumps(dict(summary=summary, episodes=res), indent=1))
    base = summary.get("ddpm-100", {}).get("success_rate")
    print("\nDPPO viability: a short chain is usable if it stays close to DDPM-100.")
    for k, s in summary.items():
        if k != "ddpm-100" and base is not None:
            print(f"  {k}: {(s['success_rate']-base)*100:+.1f} pts vs DDPM-100, "
                  f"{base and s['mean_wall']/summary['ddpm-100']['mean_wall']:.2f}x wall")


if __name__ == "__main__":
    main()

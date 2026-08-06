"""Closed-loop success-rate eval for an ACT policy on the rotate-T task.

Mirrors scripts/eval_dp_rotate_t.py (identical env, inits, controller, success criterion)
but runs ACT inference with temporal aggregation (query every step, exp-weighted ensemble of
overlapping action chunks) -- ACT's standard eval-time behavior. Directly comparable to the
Diffusion Policy numbers.

Usage:
  MUJOCO_GL=egl python scripts/eval_act_rotate_t.py \
    --ckpt outputs/act_rotate_t/policy_last.ckpt --stats outputs/act_rotate_t/stats.pkl \
    --config outputs/act_rotate_t/policy_config.pkl --n_episodes 50 --max_steps 800
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ACT_ROOT = "/home/jacobhb/projects/worth_doing/act"
sys.path.insert(0, ACT_ROOT)
sys.path.insert(0, ACT_ROOT + "/detr")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "data_collection"))

import eval_dp_rotate_t as E  # get_obs, eval_success, DT/gains, trajectory helpers
import collect_rotate_t as C
from sim_aloha_dataset_collection_scripted import trajectory_to_joint_actions
from interactive_world_sim.utils.pose_utils import PoseType, pose_convert
import gym_aloha.env as gae
from gym_aloha.env import AlohaEnv
from yixuan_utilities.kinematics_helper import KinHelper


def load_act(ckpt, stats_path, config_path, device):
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    saved = sys.argv
    sys.argv = ["x", "--ckpt_dir", "/tmp/x", "--policy_class", "ACT",
                "--task_name", "sim_rotate_t", "--seed", "0", "--num_epochs", "1"]
    from policy import ACTPolicy
    policy = ACTPolicy(cfg)
    sys.argv = saved
    policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
    policy.to(device).eval()
    return policy, cfg, stats


@torch.inference_mode()
def run_episode_act(env, policy, kin, world_t_bases, stats, chunk, max_steps, device, record):
    curr_vel = np.zeros(6)
    init_pose = E.env_state_to_mat(env._env.task.get_observation(env._env.physics)["env_state"])
    qm, qs = stats["qpos_mean"], stats["qpos_std"]
    am, as_ = stats["action_mean"], stats["action_std"]

    all_time = torch.zeros(max_steps, max_steps + chunk, 4, device=device)
    frames = []
    reached = False
    for t in range(max_steps):
        image, qpos_raw, frame = E.get_obs(env, kin, world_t_bases)
        if record:
            frames.append(frame)
        qpos = torch.from_numpy(((qpos_raw - qm) / qs).astype(np.float32)).to(device)[None]
        img = torch.from_numpy(image).to(device)[None, None]  # (1,1,3,128,128)
        a_hat = policy(qpos, img)  # (1, chunk, 4) normalized
        all_time[t, t:t + chunk] = a_hat[0]
        col = all_time[:, t]  # (max_steps, 4)
        pop = torch.all(col != 0, dim=1)
        acts = col[pop]
        w = torch.exp(-0.01 * torch.arange(len(acts), device=device))
        w = (w / w.sum())[:, None]
        raw = (acts * w).sum(0).cpu().numpy()  # normalized (4,)
        action = raw * as_ + am  # un-normalized target EE-xy

        o = env._env.task.get_observation(env._env.physics)
        joint, _ = trajectory_to_joint_actions(
            action.astype(np.float64), world_t_bases, kin, o["qpos"][:14],
            curr_vel, E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM)
        env.step(joint)
        if C.t_angle(env) <= C.TARGET_ANGLE + C.ANGLE_TOL:
            reached = True
            break
    final_pose = E.env_state_to_mat(env._env.task.get_observation(env._env.physics)["env_state"])
    return init_pose, final_pose, t + 1, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--video_dir", default="outputs/act_eval/videos")
    ap.add_argument("--n_videos", type=int, default=3)
    args = ap.parse_args()

    device = torch.device(args.device)
    policy, cfg, stats = load_act(args.ckpt, args.stats, args.config, device)
    chunk = cfg["num_queries"]
    print(f"Loaded ACT (chunk_size={chunk}) from {args.ckpt}")

    kin = KinHelper(robot_name="trossen_vx300s")
    env = AlohaEnv("pusht")
    gae.sample_pusht_pose = C.make_upright_pose_sampler((-0.08, 0.08), (-0.08, 0.08))
    if args.video_dir:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    deltas, succ, steps_used = [], [], []
    for ep in range(args.n_episodes):
        np.random.seed(args.seed + ep)  # identical inits to the DP eval
        attempts = 0
        while True:
            env.reset(seed=args.seed + ep * 1000 + attempts)
            C.stabilize_t(env)  # let the T fall/settle before observing
            o = env._env.task.get_observation(env._env.physics)
            lb = pose_convert(o["left_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            rb = pose_convert(o["right_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            world_t_bases = np.stack([lb, rb])
            C.settle_arms(env, world_t_bases, kin, np.zeros(6), E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM)
            attempts += 1
            if abs(C.t_angle(env)) <= C.UPRIGHT_TOL or attempts >= 10:
                break
        t0 = time.time()
        init_pose, final_pose, steps, frames = run_episode_act(
            env, policy, kin, world_t_bases, stats, chunk, args.max_steps, device, ep < args.n_videos)
        ok, d = E.eval_success(init_pose, final_pose)
        deltas.append(np.degrees(d)); succ.append(ok); steps_used.append(steps)
        print(f"  ep {ep:2d}: rotated {np.degrees(d):+6.1f} deg  steps={steps:3d}  "
              f"{'SUCCESS' if ok else 'fail   '}  ({time.time()-t0:.0f}s)", flush=True)
        if frames:
            import imageio.v2 as imageio
            up = np.stack([cv2.resize(f, (512, 512), interpolation=cv2.INTER_NEAREST) for f in frames])
            imageio.mimwrite(f"{args.video_dir}/act_ep{ep}_{np.degrees(d):+.0f}deg_{'ok' if ok else 'fail'}.mp4",
                             up, fps=30, codec="libx264", pixelformat="yuv420p", output_params=["-crf", "18"])

    sr = float(np.mean(succ)); n = len(succ); se = (sr * (1 - sr) / n) ** 0.5
    deltas = np.array(deltas)
    print("\n================ ACT RESULTS ================")
    print(f"episodes:      {n}")
    print(f"SUCCESS RATE:  {sr*100:.1f}%  (+/- {se*100:.1f}% SE; {sum(succ)}/{n})")
    print(f"rotation deg:  mean={deltas.mean():+.1f}  std={deltas.std():.1f}  min={deltas.min():+.1f}  max={deltas.max():+.1f}")
    print(f"all clockwise (<0): {int(np.sum(deltas<0))}/{n}")


if __name__ == "__main__":
    main()

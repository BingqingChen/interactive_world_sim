"""Closed-loop success-rate eval for a Diffusion Policy trained on the rotate-T demos.

Loads a DP checkpoint, rolls the policy out in the MuJoCo/ALOHA `pusht` sim, and measures
how often it actually rotates the (upright, random-XY) T ~90 degrees clockwise.

The policy predicts target bimanual EE-xy (the same action space the demos recorded), which
is actuated through the identical PID/IK controller used during data collection
(`trajectory_to_joint_actions`). Observations are built identically to training:
center-cropped/resized top_pov image + current bimanual EE-xy proprioception.

Usage:
  MUJOCO_GL=egl python scripts/eval_dp_rotate_t.py \
    --ckpt /home/jacobhb/projects/worth_doing/diffusion_policy/data/outputs/<...>/checkpoints/epoch=0050-val_loss=0.0226.ckpt \
    --n_episodes 25 --max_steps 300 --video_dir outputs/dp_eval/videos
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import collections
import sys
import time
from pathlib import Path

import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

DP_ROOT = "/home/jacobhb/projects/worth_doing/diffusion_policy"
sys.path.insert(0, DP_ROOT)
sys.path.insert(0, str(Path(__file__).parent / "data_collection"))

OmegaConf.register_new_resolver("eval", eval, replace=True)  # DP configs use ${eval:}

import gym_aloha.env as gae  # noqa: E402
from gym_aloha.env import AlohaEnv  # noqa: E402
from yixuan_utilities.kinematics_helper import KinHelper  # noqa: E402
from yixuan_utilities.draw_utils import center_crop  # noqa: E402

from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402

from sim_aloha_dataset_collection_scripted import (  # noqa: E402
    env_state_to_mat,
    get_current_arm_positions,
    trajectory_to_joint_actions,
)
from interactive_world_sim.utils.pose_utils import PoseType, pose_convert  # noqa: E402
import collect_rotate_t as C  # noqa: E402  (settle_arms, make_upright_pose_sampler, t_angle, consts)

from interactive_world_sim.utils.mujoco_contacts import (  # noqa: E402
    build_contact_sets,
    gripper_t_contact,
)

# controller gains, matching data collection
DT, K_P, K_V, ACC_LIM, VEL_LIM = 1 / 10.0, 50, 10, 10.0, 0.04


def load_policy(ckpt_path, device):
    # map_location="cpu": the payload holds model+EMA+optimizer (~8GB peak on GPU
    # otherwise); only the policy itself is moved to the device below.
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill,
                         weights_only=False, map_location="cpu")
    cfg = payload["cfg"]
    workspace: BaseWorkspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device).eval()
    return policy, int(cfg.n_obs_steps), int(cfg.n_action_steps)


def get_obs(env, kin_helper, world_t_bases):
    """Replicate the exact training obs: center-crop/resize top_pov + current EE-xy."""
    obs = env._env.task.get_observation(env._env.physics)
    img = obs["images"]["top_pov"]
    crop = center_crop(img, (128, 128))
    rz = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
    image = np.moveaxis(rz, -1, 0).astype(np.float32) / 255.0  # (3,128,128) in [0,1]
    agent_pos = get_current_arm_positions(obs, kin_helper, world_t_bases).astype(np.float32)
    return image, agent_pos, rz


def eval_success(init_pose, final_pose):
    table_n = np.array([0, 0, 1])
    flat = (init_pose[:3, 2] @ table_n > 0.95) and (final_pose[:3, 2] @ table_n > 0.95)
    d = np.linalg.inv(init_pose) @ final_pose
    delta = float(np.arctan2(d[1, 0], d[0, 0]))
    ok = flat and (-C.ACCEPT_MAX <= delta <= -C.ACCEPT_MIN)
    return ok, delta


@torch.no_grad()
def run_episode(env, policy, kin_helper, world_t_bases, n_obs, n_act, max_steps, record,
                contact_sets=None):
    curr_vel = np.zeros(6)
    init_pose = env_state_to_mat(
        env._env.task.get_observation(env._env.physics)["env_state"]
    )
    policy.reset()
    device = policy.device

    img0, ap0, frame0 = get_obs(env, kin_helper, world_t_bases)
    img_hist = collections.deque([img0] * n_obs, maxlen=n_obs)
    ap_hist = collections.deque([ap0] * n_obs, maxlen=n_obs)
    frames = [frame0] if record else None
    # gripper<->T contact tracking, sampled once per 10 Hz control step
    contacts = {"contact_events": 0, "contact_steps": 0,
                "left_contact_steps": 0, "right_contact_steps": 0}
    prev_touch = False

    steps, reached = 0, False
    while steps < max_steps:
        obs_dict = {
            "image": torch.from_numpy(np.stack(img_hist)[None]).to(device),  # (1,To,3,128,128)
            "agent_pos": torch.from_numpy(np.stack(ap_hist)[None]).to(device),  # (1,To,4)
        }
        action = policy.predict_action(obs_dict)["action"][0].cpu().numpy()  # (Ta,4)
        for target_xy in action:
            o = env._env.task.get_observation(env._env.physics)
            joint, _ = trajectory_to_joint_actions(
                target_xy.astype(np.float64), world_t_bases, kin_helper,
                o["qpos"][:14], curr_vel, DT, K_P, K_V, ACC_LIM, VEL_LIM,
            )
            env.step(joint)
            img, ap, fr = get_obs(env, kin_helper, world_t_bases)
            img_hist.append(img)
            ap_hist.append(ap)
            if record:
                frames.append(fr)
            if contact_sets is not None:
                in_l, in_r = gripper_t_contact(env, contact_sets)
                touch = in_l or in_r
                if touch and not prev_touch:
                    contacts["contact_events"] += 1
                prev_touch = touch
                contacts["contact_steps"] += int(touch)
                contacts["left_contact_steps"] += int(in_l)
                contacts["right_contact_steps"] += int(in_r)
            steps += 1
            if C.t_angle(env) <= C.TARGET_ANGLE + C.ANGLE_TOL:  # reached ~>=83deg CW
                reached = True
            if steps >= max_steps:
                break
        if reached:
            break

    final_pose = env_state_to_mat(
        env._env.task.get_observation(env._env.physics)["env_state"]
    )
    return init_pose, final_pose, steps, frames, contacts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_episodes", type=int, default=25)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--video_dir", default="outputs/dp_eval/videos")
    ap.add_argument("--n_videos", type=int, default=4)
    ap.add_argument("--n_action_steps", type=int, default=None,
                    help="Override receding-horizon action steps executed per re-plan (inference-time).")
    ap.add_argument("--x_min", type=float, default=-0.08)
    ap.add_argument("--x_max", type=float, default=0.08)
    ap.add_argument("--y_min", type=float, default=-0.08)
    ap.add_argument("--y_max", type=float, default=0.08)
    ap.add_argument("--no_settle", action="store_true",
                    help="Skip the random arm settle (fixed home arm start); use with a fixed "
                         "T position (--x_min==--x_max, --y_min==--y_max) for the easy fixed task.")
    ap.add_argument("--metrics_json", default=None,
                    help="If set, write per-episode metrics (deg, steps, success, gripper-T "
                         "contacts) plus the summary to this JSON file.")
    args = ap.parse_args()

    device = torch.device(args.device)
    policy, n_obs, n_act = load_policy(args.ckpt, device)
    if args.n_action_steps is not None:
        policy.n_action_steps = args.n_action_steps  # inference-time receding horizon
        n_act = args.n_action_steps
    print(f"Loaded policy (n_obs_steps={n_obs}, n_action_steps={n_act}) from {args.ckpt}")

    kin_helper = KinHelper(robot_name="trossen_vx300s")
    env = AlohaEnv("pusht")
    gae.sample_pusht_pose = C.make_upright_pose_sampler((args.x_min, args.x_max),
                                                        (args.y_min, args.y_max))
    if args.video_dir:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    contact_sets = build_contact_sets(env)
    deltas, successes, step_counts, episode_records = [], [], [], []
    trial = 0
    for ep in range(args.n_episodes):
        np.random.seed(args.seed + ep)  # deterministic settle -> identical inits across ckpts
        # reset -> settle (varied arm start, like training) -> require upright start.
        # With --no_settle the arm stays at fixed home and the T is already upright, so a
        # single reset suffices (no retry loop needed).
        while True:
            env.reset(seed=args.seed + trial)
            trial += 1
            C.stabilize_t(env)  # let the T fall/settle before observing anything
            o = env._env.task.get_observation(env._env.physics)
            lb = pose_convert(o["left_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            rb = pose_convert(o["right_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            world_t_bases = np.stack([lb, rb])
            if args.no_settle:
                break
            C.settle_arms(env, world_t_bases, kin_helper, np.zeros(6),
                          DT, K_P, K_V, ACC_LIM, VEL_LIM)
            if abs(C.t_angle(env)) <= C.UPRIGHT_TOL:
                break  # T still upright after settle

        record = ep < args.n_videos
        t0 = time.time()
        init_pose, final_pose, steps, frames, contacts = run_episode(
            env, policy, kin_helper, world_t_bases, n_obs, n_act, args.max_steps, record,
            contact_sets=contact_sets,
        )
        ok, delta = eval_success(init_pose, final_pose)
        deltas.append(np.degrees(delta))
        successes.append(ok)
        step_counts.append(steps)
        episode_records.append({
            "episode": ep, "success": bool(ok), "rotation_deg": float(np.degrees(delta)),
            "steps": int(steps), **{k: int(v) for k, v in contacts.items()},
        })
        print(f"  ep {ep:2d}: rotated {np.degrees(delta):+6.1f} deg  steps={steps:3d}  "
              f"contacts={contacts['contact_events']:2d} "
              f"(touch {contacts['contact_steps']:3d} steps)  "
              f"{'SUCCESS' if ok else 'fail   '}  ({time.time()-t0:.0f}s)")

        if record and frames is not None:
            import imageio.v2 as imageio
            up = np.stack([cv2.resize(f, (512, 512), interpolation=cv2.INTER_NEAREST)
                           for f in frames])
            imageio.mimwrite(f"{args.video_dir}/eval_ep{ep}_{np.degrees(delta):+.0f}deg_"
                             f"{'ok' if ok else 'fail'}.mp4", up, fps=30, codec="libx264",
                             pixelformat="yuv420p", output_params=["-crf", "18"])

    sr = float(np.mean(successes))
    n = len(successes)
    se = (sr * (1 - sr) / n) ** 0.5
    deltas = np.array(deltas)
    steps_arr = np.array(step_counts)
    events = np.array([r["contact_events"] for r in episode_records])
    print("\n================ RESULTS ================")
    print(f"episodes:      {n}")
    print(f"SUCCESS RATE:  {sr*100:.1f}%  (+/- {se*100:.1f}% SE; {sum(successes)}/{n})")
    print(f"rotation deg:  mean={deltas.mean():+.1f}  std={deltas.std():.1f}  "
          f"min={deltas.min():+.1f}  max={deltas.max():+.1f}")
    print(f"steps:         mean={steps_arr.mean():.0f}  (among successes: "
          f"{np.mean([s for s,k in zip(step_counts,successes) if k] or [0]):.0f})")
    print(f"contacts:      mean events={events.mean():.1f}  "
          f"mean touch steps={np.mean([r['contact_steps'] for r in episode_records]):.0f}")
    print(f"all clockwise (<0): {int(np.sum(deltas<0))}/{n}")

    if args.metrics_json:
        import json
        summary = {
            "ckpt": args.ckpt,
            "n_episodes": n,
            "success_rate": sr,
            "n_success": int(sum(successes)),
            "success_se": se,
            "rotation_deg_mean": float(deltas.mean()),
            "rotation_deg_std": float(deltas.std()),
            "steps_mean": float(steps_arr.mean()),
            "steps_mean_success_only": float(
                np.mean([s for s, k in zip(step_counts, successes) if k] or [0])
            ),
            "contact_events_mean": float(events.mean()),
            "contact_steps_mean": float(
                np.mean([r["contact_steps"] for r in episode_records])
            ),
            "protocol": {
                "seed": args.seed, "max_steps": args.max_steps,
                "no_settle": args.no_settle,
                "x_range": [args.x_min, args.x_max], "y_range": [args.y_min, args.y_max],
            },
        }
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_json, "w") as f:
            json.dump({"summary": summary, "episodes": episode_records}, f, indent=2)
        print(f"metrics written to {args.metrics_json}")


if __name__ == "__main__":
    main()

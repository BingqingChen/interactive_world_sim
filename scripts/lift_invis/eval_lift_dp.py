"""Closed-loop success-rate eval for a Diffusion Policy trained on the robosuite Lift demos.

Evaluates one checkpoint on the FIXED initial states in `lift_invis_eval_inits.npz` -- 50 in the
+/-3 cm square and 50 in the +/-10 cm square, sampled once by sample_eval_inits.py. Every policy
sees exactly the same states, which is what makes the three conditions comparable.

Runs in-process (robosuite + torch in the same py3.11 `.venv-lift`). That is only sound because
the Phase 0 fidelity gate measured this venv's renderer against the py3.8 collection renderer at
37.4 dB / 38.4 dB median -- i.e. training and evaluation pixels come from effectively the same
renderer.

The observation is built to match training byte-for-byte in construction order:
    agentview/wrist -> [::-1] (undo the OpenGL flip, as the collector stored them)
                    -> INTER_AREA resize 256 -> 128 (as build_lift_zarrs.py did)
                    -> CHW, /255
    agent_pos       -> eef_pos(3) + eef axis-angle(3) + gripper_qpos(2), the collector's _state_vec
Getting any step of that wrong puts the policy off-distribution and silently depresses every
condition's score.

Episodes start from the stored POST-SETTLE state, so the 10 dummy settle steps the collector ran
are NOT repeated here -- they are already baked into the state.

Usage:
    MUJOCO_GL=egl .venv-lift/bin/python scripts/lift_invis/eval_lift_dp.py \
        --ckpt <path>/checkpoints/latest.ckpt --label centre \
        --inits datasets/lift_invis_eval_inits.npz \
        --out outputs/lift_invis/results_centre.json --video-dir outputs/lift_invis/videos
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import collections
import json
import math
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
sys.path.insert(0, str(Path(__file__).parent))

OmegaConf.register_new_resolver("eval", eval, replace=True)  # DP configs use ${eval:}

from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402

from replay_lift import build_env, set_cube_visible  # noqa: E402

RESOLUTION = 128


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite's transform_utils.quat2axisangle, as main_lift.py copied it."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def state_vec(obs) -> np.ndarray:
    """The 8-dim proprio vector, identical to main_lift.py:_state_vec."""
    return np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ).astype(np.float32)


def obs_images(obs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(agentview_chw, wrist_chw, agentview_128_hwc_for_video), matching the training pipeline."""
    out = []
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        img = np.ascontiguousarray(obs[key][::-1])  # upright, as stored
        small = cv2.resize(img, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
        out.append((np.moveaxis(small, -1, 0).astype(np.float32) / 255.0, small))
    return out[0][0], out[1][0], out[0][1]


def load_policy(ckpt_path: str, device: str):
    # map_location="cpu": the payload holds model + EMA + optimizer; only the policy goes to GPU.
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill,
                         weights_only=False, map_location="cpu")
    cfg = payload["cfg"]
    workspace: BaseWorkspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device).eval()
    return policy, int(cfg.n_obs_steps), int(cfg.n_action_steps)


@torch.no_grad()
def run_episode(env, policy, flat_state, n_obs, max_steps, record):
    """One episode from a stored post-settle state. Returns (success, steps_to_success, frames)."""
    env.reset()  # rebuild internal buffers; the state is overwritten immediately below
    env.sim.set_state_from_flattened(np.asarray(flat_state, dtype=np.float64))
    env.sim.forward()
    obs = env._get_observations(force_update=True)

    policy.reset()
    device = policy.device

    img0, wr0, frame0 = obs_images(obs)
    ap0 = state_vec(obs)
    img_hist = collections.deque([img0] * n_obs, maxlen=n_obs)
    wr_hist = collections.deque([wr0] * n_obs, maxlen=n_obs)
    ap_hist = collections.deque([ap0] * n_obs, maxlen=n_obs)
    frames = [frame0] if record else None

    steps, success, success_step = 0, False, -1
    while steps < max_steps:
        obs_dict = {
            "image": torch.from_numpy(np.stack(img_hist)[None]).to(device),      # (1,To,3,128,128)
            "wrist_image": torch.from_numpy(np.stack(wr_hist)[None]).to(device),
            "agent_pos": torch.from_numpy(np.stack(ap_hist)[None]).to(device),   # (1,To,8)
        }
        action_chunk = policy.predict_action(obs_dict)["action"][0].cpu().numpy()  # (Ta,7)

        for act in action_chunk:
            obs, _, _, _ = env.step(np.asarray(act, dtype=np.float64).tolist())
            steps += 1
            img, wr, fr = obs_images(obs)
            img_hist.append(img)
            wr_hist.append(wr)
            ap_hist.append(state_vec(obs))
            if record:
                frames.append(fr)
            if env._check_success() and not success:
                success, success_step = True, steps
            if success or steps >= max_steps:
                break
        if success:
            break

    return success, success_step, frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True, help="condition name, e.g. centre / vis / invis")
    ap.add_argument("--inits", type=Path, default=Path("datasets/lift_invis_eval_inits.npz"))
    ap.add_argument("--regions", nargs="+", default=["small", "large"])
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video-dir", type=Path, default=None)
    ap.add_argument("--n-videos", type=int, default=5, help="episodes to record per region")
    args = ap.parse_args()

    inits = np.load(args.inits)
    policy, n_obs, n_act = load_policy(args.ckpt, args.device)
    print(f"loaded {args.label}: n_obs_steps={n_obs} n_action_steps={n_act} from {args.ckpt}")

    env = build_env(max_steps=args.max_steps)
    set_cube_visible(env, True)  # evaluation always shows the cube, whatever the policy trained on

    results = {"label": args.label, "ckpt": str(args.ckpt), "max_steps": args.max_steps,
               "n_obs_steps": n_obs, "n_action_steps": n_act, "regions": {}}

    for region in args.regions:
        states = inits[f"{region}_sim_state"][: args.n_episodes]
        cube_xy = inits[f"{region}_cube_xy"][: args.n_episodes]
        succ, steps_to = [], []
        t0 = time.time()
        for i, st in enumerate(states):
            record = args.video_dir is not None and i < args.n_videos
            ok, sstep, frames = run_episode(env, policy, st, n_obs, args.max_steps, record)
            succ.append(bool(ok))
            steps_to.append(int(sstep))
            if record and frames:
                args.video_dir.mkdir(parents=True, exist_ok=True)
                import imageio
                imageio.mimwrite(
                    args.video_dir / f"{args.label}_{region}_ep{i:02d}_"
                                     f"{'success' if ok else 'failure'}.mp4",
                    frames, fps=20,
                )
            print(f"  {region} ep{i:02d}: success={ok} step={sstep} "
                  f"cube_xy=({cube_xy[i][0]:+.3f},{cube_xy[i][1]:+.3f})", flush=True)

        n = len(succ)
        k = int(np.sum(succ))
        rate = k / n
        sem = math.sqrt(rate * (1 - rate) / n) if n else 0.0
        results["regions"][region] = {
            "n": n, "successes": k, "success_rate": rate, "sem": sem,
            "success_flags": succ, "steps_to_success": steps_to,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{args.label} / {region}: {k}/{n} = {100 * rate:.1f}% +/- {100 * sem:.1f}%  "
              f"({time.time() - t0:.0f}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

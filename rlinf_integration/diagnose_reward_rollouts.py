"""Diagnostic: drive IWSRotateTWorldEnv with RANDOM actions (the untrained-RLPD-actor
regime -- exactly when the hardened reward readout matters most, since exploration
visits action distributions the WM never saw from the DP-driven collection scripts)
for N independent rollouts, and render each with the proxy angle/reward AND the
accept/reject decision overlaid per chunk, for visual verification of the hardening
logic (confidence-weighted median + area sanity + hard IoU floor + jump rejection).

Bypasses RLinf's Worker/Ray scheduler entirely (constructs IWSRotateTWorldEnv directly
via object.__new__ + manual base-attribute setup) -- this is a standalone reward-logic
check, not a full RLinf rollout-worker test (that's the separate end-to-end smoke test).

Also monkey-patches env.reset to a no-op AFTER the initial reset, so chunk_step's
internal batch-synchronous auto-reset (correct for real training) doesn't truncate
whichever rows finish early during recording -- we want each row's full
wm_max_chunks-length trajectory for inspection, not RLPD's actual training dynamics.

Usage:
  MUJOCO_GL=egl /home/jacobhb/RLinf/.venv/bin/python \
      rlinf_integration/diagnose_reward_rollouts.py --n_episodes 20
"""
import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from world_model_iws_rotate_t_env import IWSRotateTWorldEnv, N_OBS  # noqa: E402


def build_env(num_envs, seed_base, device):
    cfg = OmegaConf.create({
        "n_action_steps": 8,
        "wm_max_chunks": 40,
        "jump_reject_deg": 47.5,
        "min_iou": 0.5,
        "terminal_deg": -80.0,
        "dec_infer_steps": 2,
        "x_range": [-0.06, 0.06],
        "y_range": [-0.06, 0.06],
        "env_seed_base": seed_base,
    })
    env = object.__new__(IWSRotateTWorldEnv)
    env.cfg = cfg
    env.device = torch.device(device)
    env.num_envs = num_envs
    env.record_metrics = True
    env.auto_reset = True
    env.ignore_terminations = False
    env.use_rel_reward = False
    env._is_start = True
    env._elapsed_steps = 0
    env.video_cfg = None
    env.prev_step_reward = torch.zeros(num_envs, dtype=torch.float32, device=env.device)
    env.enable_kir = True
    env.dataset = env._build_dataset(cfg)
    env._init_metrics()
    return env


def label(img, lines, color):
    out = img.copy()
    for i, line in enumerate(lines):
        cv2.putText(out, line, (6, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--out_dir", default=str(
        Path("/home/jacobhb/projects/worth_doing/interactive_world_sim/outputs/rlpd_reward_diagnostic")
    ))
    ap.add_argument("--seed_base", type=int, default=850_000)  # disjoint from prior collection/eval seeds
    ap.add_argument("--action_seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building env ({args.n_episodes} rows) ...", flush=True)
    env = build_env(args.n_episodes, args.seed_base, args.device)
    obs, _ = env.reset()
    # Freeze recording: don't let chunk_step's internal auto-reset truncate a row's
    # trajectory just because another row in the batch finished first.
    env.reset = lambda *a, **k: (env._wrap_obs(), {})

    # Random actions within the WM's own fitted action-normalizer range -- the most
    # representative "arbitrary but plausible-scale" distribution to stress-test with,
    # and exactly the kind of action an untrained/early-training RLPD actor would emit.
    stats = env.wm.normalizer["action"].params_dict["action"]
    a_min = stats["input_stats"]["min"].cpu().numpy().reshape(-1)
    a_max = stats["input_stats"]["max"].cpu().numpy().reshape(-1)
    print(f"action range from WM normalizer: min={a_min} max={a_max}", flush=True)
    rng = np.random.default_rng(args.action_seed)

    B, n_act, wm_max_chunks = env.num_envs, env.n_act, env.wm_max_chunks
    frames_log = [[] for _ in range(B)]  # per-row list of (frame_u8, angle_deg, reward, accepted, reject_reason)

    for step in range(wm_max_chunks):
        actions = rng.uniform(a_min, a_max, size=(B, n_act, 4)).astype(np.float32)
        prev_lost, prev_morph, prev_jump = env._lost_track_count, env._morph_reject_count, env._jump_reject_count
        obs_list, rewards, terms, truncs, infos_list = env.chunk_step(torch.from_numpy(actions))
        # per-row bookkeeping: pull the last decoded frame + this chunk's angle/reward
        # straight off env's own post-step state (angle_prev already updated in-place).
        curr_img = (env.curr_img.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        reward_last = rewards[:, -1].numpy()
        term_last = terms[:, -1].numpy()
        angle_deg = np.degrees(env.angle_prev.numpy())
        for b in range(B):
            reason = ""
            if env._lost_track_count > prev_lost:
                reason = "LOST"
            elif env._morph_reject_count > prev_morph:
                reason = "MORPH-REJECT"
            elif env._jump_reject_count > prev_jump:
                reason = "JUMP-REJECT"
            frames_log[b].append((
                curr_img[b].copy(), float(angle_deg[b]), float(reward_last[b]),
                reason == "", reason, bool(term_last[b]),
            ))
        if step % 10 == 0:
            print(f"  step {step}/{wm_max_chunks}  lost={env._lost_track_count} "
                  f"morph_reject={env._morph_reject_count} jump_reject={env._jump_reject_count}",
                  flush=True)

    print(f"lost_track={env._lost_track_count}  morph_reject={env._morph_reject_count}  "
          f"jump_reject={env._jump_reject_count}  (out of {B * wm_max_chunks} chunk-reads)")

    for b in range(B):
        vid = []
        for (img, ang, rew, accepted, reason, term) in frames_log[b]:
            f = cv2.resize(img, (384, 384), interpolation=cv2.INTER_NEAREST)
            color = (0, 255, 0) if accepted else (0, 0, 255)
            lines = [f"angle={ang:+.1f} deg   reward={rew:+.2f}"]
            if not accepted:
                lines.append(f"REJECTED: {reason}")
            if term:
                lines.append("TERMINATED (success)")
            vid.append(label(f, lines, color))
        out_path = out_dir / f"rollout_ep{b:02d}.mp4"
        imageio.mimwrite(out_path, np.stack(vid), fps=8, codec="libx264",
                         pixelformat="yuv420p", output_params=["-crf", "20"])
    print(f"\nWrote {B} rollout videos to {out_dir}/")


if __name__ == "__main__":
    main()

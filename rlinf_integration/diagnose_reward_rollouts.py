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
        "min_iou": 0.35,
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
    ap.add_argument("--action_mode", choices=["iid_uniform", "smooth_walk"], default="smooth_walk",
                    help="iid_uniform: i.i.d. uniform over the WM normalizer's full fitted "
                         "range (worst-case OOD stress test, the original diagnostic). "
                         "smooth_walk: a Gaussian random walk in chunk-target space, step "
                         "size = the REAL per-chunk action-delta std measured from "
                         "datasets/scaling_v3/real/{A,B,C,D}.zarr, with within-chunk linear "
                         "interpolation -- approximates what an early (still mostly random "
                         "but temporally-smooth) SAC/RLPD Gaussian policy would actually emit, "
                         "rather than i.i.d. jumps to random extremes every chunk.")
    args = ap.parse_args()

    # Empirically measured (not guessed) from real trajectories: interactive_world_sim
    # scripts/analysis, see plan doc. Per-dim per-STEP delta std; the per-chunk (8-step)
    # step size used below is this * sqrt(8), i.e. real chunk-to-chunk target variability.
    REAL_STEP_DELTA_STD = np.array([0.00443962, 0.00653751, 0.00439547, 0.00649021], dtype=np.float32)
    REAL_CHUNK_DELTA_STD = REAL_STEP_DELTA_STD * np.sqrt(8)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building env ({args.n_episodes} rows) ...", flush=True)
    env = build_env(args.n_episodes, args.seed_base, args.device)
    env._debug_log = []  # opt-in: record raw (angle,iou,area) for every rejected read
    obs, _ = env.reset()
    # Freeze recording: don't let chunk_step's internal auto-reset truncate a row's
    # trajectory just because another row in the batch finished first.
    env.reset = lambda *a, **k: (env._wrap_obs(), {})

    stats = env.wm.normalizer["action"].get_input_stats()
    a_min = stats["min"].detach().cpu().numpy().reshape(-1)
    a_max = stats["max"].detach().cpu().numpy().reshape(-1)
    print(f"action range from WM normalizer: min={a_min} max={a_max}", flush=True)
    print(f"action_mode={args.action_mode}"
          + (f"  chunk_delta_std={REAL_CHUNK_DELTA_STD}" if args.action_mode == "smooth_walk" else ""),
          flush=True)
    rng = np.random.default_rng(args.action_seed)

    B, n_act, wm_max_chunks = env.num_envs, env.n_act, env.wm_max_chunks
    frames_log = [[] for _ in range(B)]  # per-row list of (frame_u8, angle_deg, reward, accepted, reject_reason)

    # smooth_walk state: per-row current chunk-target, initialized at each row's real
    # starting EE-xy (env.curr_state right after reset == home_xy).
    walk_target = env.curr_state.detach().cpu().numpy().copy() if args.action_mode == "smooth_walk" else None

    for step in range(wm_max_chunks):
        if args.action_mode == "iid_uniform":
            actions = rng.uniform(a_min, a_max, size=(B, n_act, 4)).astype(np.float32)
        else:
            prev_target = walk_target.copy()
            step_noise = rng.normal(0.0, REAL_CHUNK_DELTA_STD, size=(B, 4)).astype(np.float32)
            walk_target = np.clip(prev_target + step_noise, a_min, a_max)
            # linear interpolation across the 8 sub-steps -- no sub-step teleports,
            # matching how a real chunk-level action head outputs a smooth trajectory.
            alphas = np.linspace(1.0 / n_act, 1.0, n_act, dtype=np.float32)  # (n_act,)
            actions = (prev_target[:, None, :] +
                      alphas[None, :, None] * (walk_target - prev_target)[:, None, :]).astype(np.float32)
        prev_lost, prev_morph, prev_jump = env._lost_track_count, env._morph_reject_count, env._jump_reject_count
        obs_list, rewards, terms, truncs, infos_list = env.chunk_step(torch.from_numpy(actions))
        # per-row bookkeeping: pull the last decoded frame + this chunk's angle/reward
        # straight off env's own post-step state (angle_prev already updated in-place).
        curr_img = (env.curr_img.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        reward_last = rewards[:, -1].cpu().numpy()
        term_last = terms[:, -1].cpu().numpy()
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
                  f"morph_reject={env._morph_reject_count} jump_reject={env._jump_reject_count} "
                  f"stuck_recovery={env._stuck_recovery_count}",
                  flush=True)

    n_accepted = B * wm_max_chunks - env._lost_track_count - env._morph_reject_count - env._jump_reject_count
    print(f"lost_track={env._lost_track_count}  morph_reject={env._morph_reject_count}  "
          f"jump_reject={env._jump_reject_count}  stuck_recovery={env._stuck_recovery_count}  "
          f"accepted={n_accepted}  (out of {B * wm_max_chunks} chunk-reads)")

    # Summarize what actually drove each rejection -- the IoU/area of every candidate
    # frame, not just the pass/fail counts, so a too-strict threshold is visible directly.
    morph_ious = []  # best IoU seen among the (failed) candidates of each morph-reject event
    for ev in env._debug_log:
        if ev["kind"] == "morph":
            best_iou = max((iou for (_a, iou, _ar) in ev["raw"]), default=0.0)
            morph_ious.append(best_iou)
    if morph_ious:
        morph_ious = np.array(morph_ious)
        print(f"morph_reject best-IoU-seen distribution: n={len(morph_ious)} "
              f"mean={morph_ious.mean():.3f} median={np.median(morph_ious):.3f} "
              f"min={morph_ious.min():.3f} max={morph_ious.max():.3f} "
              f"(all are just below min_iou={env.min_iou} by definition of this bucket "
              f"unless area check was the actual blocker)")
        # how many would flip to accepted at a few candidate lower thresholds
        for thresh in (0.45, 0.40, 0.35, 0.30):
            n_would_pass = int((morph_ious >= thresh).sum())
            print(f"  if min_iou were {thresh}: {n_would_pass}/{len(morph_ious)} of these "
                  f"morph-rejects would have passed on their best candidate frame")
    jump_events = [ev for ev in env._debug_log if ev["kind"] == "jump"]
    if jump_events:
        jd = np.array([ev["jump_deg"] for ev in jump_events])
        print(f"jump_reject jump_deg distribution: n={len(jd)} mean={jd.mean():.1f} "
              f"median={np.median(jd):.1f} min={jd.min():.1f} max={jd.max():.1f}")

    import json
    debug_path = out_dir / "debug_log.json"
    with open(debug_path, "w") as f:
        json.dump([
            {**{k: v for k, v in ev.items() if k != "raw"},
             "raw": [[a, iou, area] for (a, iou, area) in ev["raw"]]}
            for ev in env._debug_log
        ], f, indent=1)
    print(f"Full per-rejection debug log: {debug_path}")

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
    env.close()


if __name__ == "__main__":
    main()

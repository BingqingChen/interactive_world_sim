"""Render a sample of full episodes from the existing imagined pool
(datasets/scaling_v3/imag/{half1,half2}.zarr) as labeled videos, for visual
inspection of the CURRENT reward/reject/morph-detect status -- per user
request, after: (a) the full-800-episode reward-heuristic validation,
(b) the morph_terminate_patience=3 fix, (c) the reward fix (prog=0/no success
on morph_detected chunks), and (d) the finding that area-ratio-based
morph_detected misses at least one real hallucination mode (a T arm
disappearing while area stays ~1.0, e.g. half1 ep1 frame 159) that a low
best-match IoU (median ~0.52 for jump-reject frames vs ~0.82 pool-wide) does
catch.

Reuses the REAL production _robust_angle_end (via validate_reward_heuristics_
full_pool.build_thresholded_env/process_episode) for angle/accept/morph_detected,
then separately recomputes best_iou/area_ratio for EVERY chunk (not just
rejects, which is all _robust_angle_end's own debug_log covers) so the video
overlay can show them continuously. Reward is computed with the CURRENT
(fixed) formula: prog=0 and success=False whenever morph_detected, angle_prev
frozen on morph_detected chunks -- mirroring world_model_iws_rotate_t_env.py's
chunk_step exactly (kept in sync by hand; if that file's formula changes,
update the REWARD block below too).

Renders the FULL stored episode (all raw frames), not a truncated preview.

Usage:
  MUJOCO_GL=egl /home/jacobhb/RLinf/.venv/bin/python \
      rlinf_integration/render_imagined_pool_sample_videos.py \
      --n_episodes 20 --seed 0 \
      --out_dir outputs/imagined_pool_sample_videos
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import zarr

WORKTREE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE_ROOT / "rlinf_integration"))
sys.path.insert(0, str(WORKTREE_ROOT / "scripts"))
sys.path.insert(0, str(WORKTREE_ROOT / "scripts" / "data_collection"))

from validate_reward_heuristics_full_pool import build_thresholded_env  # noqa: E402
from collect_imagined_rotate_t import make_templates, est_angle_with_conf, red_mask  # noqa: E402
from world_model_iws_rotate_t_env import REWARD_THETAS  # noqa: E402


def label(img, lines, color):
    out = img.copy()
    for i, line in enumerate(lines):
        cv2.putText(out, line, (6, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    return out


def process_episode_full(env, imgs, n_act, morph_terminate_patience):
    """Like validate_reward_heuristics_full_pool.process_episode, but also
    computes best_iou/area_ratio for EVERY chunk (not just rejects) and the
    CURRENT reward formula (prog=0/no success on morph_detected, angle_prev
    frozen), plus the patience counter -- everything the video overlay needs."""
    L = len(imgs)
    frame0_u8 = imgs[0]
    templates, tc = make_templates(frame0_u8, thetas=REWARD_THETAS)
    f0_area = int(red_mask(frame0_u8).sum())

    env._templates = [templates]
    env._tc = [tc]
    env._frame0_area = [f0_area]
    env.angle_prev = np.zeros(1, dtype=np.float64)
    env._consec_reject = np.zeros(1, dtype=np.int64)
    env._jump_reject_count = 0
    env._lost_track_count = 0
    env._morph_reject_count = 0
    env._stuck_recovery_count = 0
    env._debug_log = []

    consec_morph = 0
    n_chunks = L // n_act
    per_chunk = []
    for c in range(1, n_chunks + 1):
        end = c * n_act
        k = min(3, n_act)
        last3 = [imgs[j] for j in range(end - k, end)]
        prev_before = float(env.angle_prev[0])
        log_len_before = len(env._debug_log)
        angle_end, accepted, morph_detected = env._robust_angle_end(last3, 0)
        debug_kind = None
        if len(env._debug_log) > log_len_before:
            debug_kind = env._debug_log[-1]["kind"]

        # Supplementary: best_iou/area_ratio of the LAST frame of the chunk,
        # for continuous display (not just on reject). Independent, read-only
        # re-computation -- doesn't affect env state.
        last_frame = imgs[end - 1]
        _angle_last, best_iou_last, area_last = est_angle_with_conf(
            last_frame, templates, tc, thetas=REWARD_THETAS)
        area_ratio_last = area_last / f0_area if f0_area else 0.0

        # CURRENT reward formula (mirrors world_model_iws_rotate_t_env.py's
        # chunk_step, kept in sync by hand -- see module docstring).
        if morph_detected:
            prog = 0.0
            success = False
        else:
            prog = prev_before - angle_end
            success = bool(angle_end <= np.radians(env.terminal_deg))
            env.angle_prev[0] = angle_end
        reward = prog * 5.0 - 0.01 + (5.0 if success else 0.0)

        if morph_detected:
            consec_morph += 1
        else:
            consec_morph = 0
        morph_terminate = consec_morph >= morph_terminate_patience

        per_chunk.append(dict(
            chunk=c, end_frame=end, prev_deg=np.degrees(prev_before),
            angle_end_deg=np.degrees(angle_end), accepted=accepted,
            morph_detected=bool(morph_detected), debug_kind=debug_kind,
            best_iou_last_frame=float(best_iou_last), area_ratio_last_frame=float(area_ratio_last),
            reward=float(reward), success=bool(success),
            consec_morph=int(consec_morph), morph_terminate=bool(morph_terminate),
        ))
    return per_chunk, f0_area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarrs", nargs="+", default=[
        "datasets/scaling_v3/imag/half1.zarr", "datasets/scaling_v3/imag/half2.zarr"])
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="outputs/imagined_pool_sample_videos")
    args = ap.parse_args()

    env = build_thresholded_env()
    n_act = env.n_act
    patience = env.morph_terminate_patience
    print(f"n_act={n_act} morph_terminate_patience={patience} jump_reject_deg={env.jump_reject_deg} "
          f"area_ratio=[{env.area_ratio_low},{env.area_ratio_high}] min_iou={env.min_iou} "
          f"terminal_deg={env.terminal_deg}")

    # Build a flat (zarr_path, episode_idx) index across all pools, then sample.
    all_eps = []
    zarr_cache = {}
    for zpath in args.zarrs:
        z = zarr.open(zpath, mode="r")
        zarr_cache[zpath] = z
        ends = z["meta/episode_ends"][:]
        for ep in range(len(ends)):
            all_eps.append((zpath, ep))
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(all_eps), size=min(args.n_episodes, len(all_eps)), replace=False)
    picks = sorted(picks.tolist())
    print(f"Sampled {len(picks)} episodes (seed={args.seed}) out of {len(all_eps)} total")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for rank, pick_idx in enumerate(picks):
        zpath, ep = all_eps[pick_idx]
        z = zarr_cache[zpath]
        ends = z["meta/episode_ends"][:]
        starts = np.concatenate([[0], ends[:-1]])
        s, e = int(starts[ep]), int(ends[ep])
        imgs = z["data/img"][s:e]  # (L,128,128,3) uint8, FULL episode, no truncation
        L = len(imgs)
        if L < n_act:
            continue

        per_chunk, f0_area = process_episode_full(env, imgs, n_act, patience)

        zname = Path(zpath).stem
        ep_tag = f"{zname}_ep{ep:03d}"
        print(f"[{rank+1}/{len(picks)}] {ep_tag}  L={L} frames ({L//n_act} chunks)  "
              f"morph_terminate_first_hit={'chunk ' + str(next((c['chunk'] for c in per_chunk if c['morph_terminate']), None)) if any(c['morph_terminate'] for c in per_chunk) else 'never'}")

        # Render EVERY raw frame of the FULL episode -- not a truncated preview.
        vid = []
        chunk_idx = 0
        cur = None
        for t in range(L):
            if t % n_act == 0 and chunk_idx < len(per_chunk):
                cur = per_chunk[chunk_idx]
                chunk_idx += 1
            f = cv2.resize(imgs[t], (384, 384), interpolation=cv2.INTER_NEAREST)
            if cur is None:
                lines = [f"frame {t}  (pre-first-chunk)"]
                color = (255, 255, 255)
            else:
                status = "REJECTED:" + cur["debug_kind"] if not cur["accepted"] else (
                    "accepted (stuck_recovery)" if cur["debug_kind"] == "stuck_recovery" else "accepted")
                color = (0, 0, 255) if not cur["accepted"] else (
                    (0, 165, 255) if cur["morph_detected"] else (0, 255, 0))
                lines = [
                    f"frame {t}  chunk {cur['chunk']}/{len(per_chunk)}  angle={cur['angle_end_deg']:+.1f}deg",
                    f"reward={cur['reward']:+.2f}  {status}",
                    f"best_iou={cur['best_iou_last_frame']:.3f}  area_ratio={cur['area_ratio_last_frame']:.2f}",
                ]
                if cur["morph_detected"]:
                    lines.append(f"MORPH_DETECTED  (consec={cur['consec_morph']}/{patience})")
                if cur["morph_terminate"]:
                    lines.append(">>> morph_terminate WOULD FIRE HERE <<<")
                if cur["success"]:
                    lines.append("SUCCESS (terminal)")
            vid.append(label(f, lines, color))
        out_path = out_dir / f"{ep_tag}.mp4"
        imageio.mimwrite(out_path, np.stack(vid), fps=8, codec="libx264",
                         pixelformat="yuv420p", output_params=["-crf", "20"])

        summary.append(dict(
            zarr=zpath, episode=int(ep), n_frames=L, n_chunks=len(per_chunk),
            n_accepted=sum(1 for c in per_chunk if c["accepted"]),
            n_morph_detected=sum(1 for c in per_chunk if c["morph_detected"]),
            n_jump_reject=sum(1 for c in per_chunk if c["debug_kind"] == "jump"),
            n_lost_track=sum(1 for c in per_chunk if c["debug_kind"] == "lost"),
            n_stuck_recovery=sum(1 for c in per_chunk if c["debug_kind"] == "stuck_recovery"),
            morph_terminate_first_chunk=next((c["chunk"] for c in per_chunk if c["morph_terminate"]), None),
            any_success=any(c["success"] for c in per_chunk),
            total_reward=sum(c["reward"] for c in per_chunk),
            video=str(out_path),
        ))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(summary)} videos + summary.json to {out_dir}/")


if __name__ == "__main__":
    main()

"""Final out-of-sample check of the WM reward estimator's reject/recovery logic,
run against the FULL 800-episode imagined pool (datasets/scaling_v3/imag/{half1,half2}.zarr),
per user request: "confirm 1) the jump reject, and 2) the morph reject rule are
flagging correctly. And there is correct mechanism to either recover or early stop
when hallucination persists."

Reuses the REAL production code (IWSRotateTWorldEnv._robust_angle_end) as an unbound
method against a lightweight stand-in object, rather than reimplementing the logic --
avoids the exact "silent shadowing" bug class already found once (an earlier
diagnostic script's build_env() hardcoded min_iou=0.5, silently shadowing the env's
own tuned default). Thresholds themselves are extracted by actually calling
_build_dataset({}) (with the WM/subprocess parts mocked out), not copy-pasted, so
there is zero drift risk if the class's defaults ever change.

Each episode's own frame 0 (already real-reset-grounded when the pool was
originally collected) is used to build that episode's template library, exactly
matching what a live env's reset() does -- so this reproduces the EXACT reward path
a real rollout would see, just applied to already-rendered frames instead of live
WM inference (which needs no GPU here: est_angle_with_conf is pure OpenCV).

For the recovery/early-stop question, this script computes morph_detected for
EVERY chunk in each episode's full (uninterrupted) length -- independent of what
the production early-terminate policy would have done -- so it can measure true
run-length statistics (a transient blip vs. sustained hallucination) and compare
two termination policies against the same ground truth:
  - "immediate" (current production behavior, world_model_iws_rotate_t_env.py:475):
    truncate at the FIRST morph_detected chunk.
  - "patience=N": truncate only after N consecutive morph_detected chunks.

Usage:
  MUJOCO_GL=egl /home/jacobhb/RLinf/.venv/bin/python \
      rlinf_integration/validate_reward_heuristics_full_pool.py \
      --zarrs datasets/scaling_v3/imag/half1.zarr datasets/scaling_v3/imag/half2.zarr \
      --patience 3 --out_dir outputs/reward_heuristics_full_pool_validation
"""
import argparse
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import zarr

WORKTREE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE_ROOT / "rlinf_integration"))
sys.path.insert(0, str(WORKTREE_ROOT / "scripts"))
sys.path.insert(0, str(WORKTREE_ROOT / "scripts" / "data_collection"))

import world_model_iws_rotate_t_env as WM_ENV_MOD  # noqa: E402
from world_model_iws_rotate_t_env import IWSRotateTWorldEnv, REWARD_THETAS  # noqa: E402
from collect_imagined_rotate_t import make_templates, red_mask  # noqa: E402


def build_thresholded_env():
    """Get a bare IWSRotateTWorldEnv with real thresholds set via the REAL
    _build_dataset({}) code path (WM load / real-reset subprocess mocked out --
    neither is needed for _robust_angle_end, which is pure OpenCV/numpy)."""
    env = object.__new__(IWSRotateTWorldEnv)
    with mock.patch.object(WM_ENV_MOD.M, "load_viz_cfg", return_value=mock.MagicMock()), \
         mock.patch.object(WM_ENV_MOD.M, "load_model", return_value=mock.MagicMock()), \
         mock.patch.object(WM_ENV_MOD.subprocess, "Popen", return_value=mock.MagicMock()), \
         mock.patch.object(IWSRotateTWorldEnv, "_get_runtime_device_str", return_value="cpu"):
        env._build_dataset({})
    return env


def process_episode(env, imgs, n_act):
    """Run the real per-chunk reward/reject logic over one episode's full,
    already-rendered frame sequence. Returns a list of per-chunk dicts."""
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

    n_chunks = L // n_act
    per_chunk = []
    for c in range(1, n_chunks + 1):
        end = c * n_act
        k = min(3, n_act)
        last3 = [imgs[j] for j in range(end - k, end)]
        prev_before = float(env.angle_prev[0])
        angle_end, accepted, morph_detected = env._robust_angle_end(last3, 0)
        log_before = len(env._debug_log)
        per_chunk.append(dict(
            chunk=c, end_frame=end, prev_deg=np.degrees(prev_before),
            angle_end_deg=np.degrees(angle_end), accepted=accepted,
            morph_detected=morph_detected,
            debug_kind=(env._debug_log[-1]["kind"] if len(env._debug_log) > log_before - 1 and env._debug_log else None),
        ))
        env.angle_prev[0] = angle_end
    return per_chunk, dict(
        jump_reject_count=env._jump_reject_count,
        lost_track_count=env._lost_track_count,
        morph_reject_count=env._morph_reject_count,
        stuck_recovery_count=env._stuck_recovery_count,
        debug_log=env._debug_log,
    )


def run_length_stats(flags):
    """flags: list[bool]. Returns list of (start_idx, length) for each maximal
    True-run."""
    runs = []
    i = 0
    n = len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            runs.append((i, j - i))
            i = j
        else:
            i += 1
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarrs", nargs="+", default=[
        "datasets/scaling_v3/imag/half1.zarr", "datasets/scaling_v3/imag/half2.zarr"])
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--out_dir", default="outputs/reward_heuristics_full_pool_validation")
    ap.add_argument("--n_visual_morph", type=int, default=24)
    ap.add_argument("--n_visual_jump", type=int, default=24)
    ap.add_argument("--max_episodes", type=int, default=None, help="per zarr, for a quick smoke test")
    args = ap.parse_args()

    env = build_thresholded_env()
    n_act = env.n_act
    print(f"Thresholds (from real _build_dataset defaults): jump_reject_deg={env.jump_reject_deg}, "
          f"area_ratio=[{env.area_ratio_low},{env.area_ratio_high}], min_iou={env.min_iou}, "
          f"stuck_reject_limit={env.stuck_reject_limit}, morph_terminate_patience={env.morph_terminate_patience}, "
          f"n_act={n_act}, terminal_deg={env.terminal_deg}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_episode_results = []  # for JSON dump
    agg = dict(jump_reject_count=0, lost_track_count=0, morph_reject_count=0,
               stuck_recovery_count=0, total_chunks=0, total_episodes=0)
    morph_run_lengths = []
    jump_events = []   # (zarr, ep, chunk, jump_deg, ...)
    morph_events = []  # (zarr, ep, chunk, ...)
    immediate_terminate_points = []  # chunk index of first morph per episode (or None)
    patience_terminate_points = []
    recovered_after_isolated = 0
    isolated_morph_count = 0

    for zpath in args.zarrs:
        z = zarr.open(zpath, mode="r")
        ends = z["meta/episode_ends"][:]
        starts = np.concatenate([[0], ends[:-1]])
        n_eps = len(ends) if args.max_episodes is None else min(len(ends), args.max_episodes)
        print(f"\n{zpath}: {n_eps} episodes")
        for ep in range(n_eps):
            s, e = int(starts[ep]), int(ends[ep])
            imgs = z["data/img"][s:e]
            if len(imgs) < n_act:
                continue
            per_chunk, ep_counts = process_episode(env, imgs, n_act)
            agg["jump_reject_count"] += ep_counts["jump_reject_count"]
            agg["lost_track_count"] += ep_counts["lost_track_count"]
            agg["morph_reject_count"] += ep_counts["morph_reject_count"]
            agg["stuck_recovery_count"] += ep_counts["stuck_recovery_count"]
            agg["total_chunks"] += len(per_chunk)
            agg["total_episodes"] += 1

            morph_flags = [c["morph_detected"] for c in per_chunk]
            runs = run_length_stats(morph_flags)
            morph_run_lengths.extend([r[1] for r in runs])

            # Immediate-terminate policy: first morph chunk.
            first_morph = next((c["chunk"] for c in per_chunk if c["morph_detected"]), None)
            immediate_terminate_points.append(first_morph)

            # Patience=N policy: first chunk where a run of >=N consecutive
            # morph_detected has just completed.
            patience_point = None
            for (start_idx, length) in runs:
                if length >= args.patience:
                    patience_point = per_chunk[start_idx + args.patience - 1]["chunk"]
                    break
            patience_terminate_points.append(patience_point)

            # Recovery check: for isolated runs (length < patience) that are NOT
            # the episode's last chunks, check the next `patience` chunks are clean.
            for (start_idx, length) in runs:
                if length < args.patience and (start_idx + length + args.patience) <= len(per_chunk):
                    isolated_morph_count += 1
                    followup = morph_flags[start_idx + length: start_idx + length + args.patience]
                    if not any(followup):
                        recovered_after_isolated += 1

            # collect events for visual sampling
            for c in per_chunk:
                if c["debug_kind"] == "jump":
                    jump_events.append((zpath, ep, c["chunk"], c["end_frame"], c["prev_deg"], c["angle_end_deg"]))
                if c["morph_detected"]:
                    morph_events.append((zpath, ep, c["chunk"], c["end_frame"]))

            all_episode_results.append(dict(
                zarr=zpath, episode=int(ep), n_chunks=len(per_chunk),
                first_morph_chunk=first_morph, patience_terminate_chunk=patience_point,
                morph_run_lengths=[r[1] for r in runs],
                n_chunks_accepted=sum(1 for c in per_chunk if c["accepted"]),
            ))
            if ep % 100 == 0:
                print(f"  ep{ep:03d} done ({len(per_chunk)} chunks)")

    print("\n" + "=" * 70)
    print("AGGREGATE (all episodes, all zarrs)")
    print("=" * 70)
    print(json.dumps(agg, indent=2))
    total_checks = agg["total_chunks"]
    print(f"\naccept rate: {100*(1 - (agg['jump_reject_count']+agg['lost_track_count']+agg['morph_reject_count'])/max(total_checks,1)):.2f}%")
    print(f"jump_reject rate: {100*agg['jump_reject_count']/max(total_checks,1):.3f}%")
    print(f"morph (IoU-floor) reject rate: {100*agg['morph_reject_count']/max(total_checks,1):.3f}%")
    print(f"lost_track rate: {100*agg['lost_track_count']/max(total_checks,1):.3f}%")
    print(f"stuck_recovery events: {agg['stuck_recovery_count']}")

    n_eps = agg["total_episodes"]
    n_immediate_term = sum(1 for p in immediate_terminate_points if p is not None)
    n_patience_term = sum(1 for p in patience_terminate_points if p is not None)
    print(f"\nEpisodes with >=1 morph_detected chunk (immediate policy would truncate): "
          f"{n_immediate_term}/{n_eps} ({100*n_immediate_term/n_eps:.1f}%)")
    print(f"Episodes that reach a run of >={args.patience} consecutive morph_detected "
          f"chunks (patience={args.patience} policy would truncate): "
          f"{n_patience_term}/{n_eps} ({100*n_patience_term/n_eps:.1f}%)")
    n_saved_by_patience = n_immediate_term - n_patience_term
    print(f"Episodes 'saved' by patience={args.patience} vs immediate: {n_saved_by_patience} "
          f"({100*n_saved_by_patience/max(n_immediate_term,1):.1f}% of immediate-terminated episodes)")

    if morph_run_lengths:
        arr = np.array(morph_run_lengths)
        print(f"\nmorph_detected run-length distribution (n={len(arr)} runs): "
              f"mean={arr.mean():.2f} median={np.median(arr):.1f} max={arr.max()} "
              f"p90={np.percentile(arr,90):.1f} frac_len1={100*np.mean(arr==1):.1f}% "
              f"frac_len_ge_{args.patience}={100*np.mean(arr>=args.patience):.1f}%")

    if isolated_morph_count > 0:
        print(f"\nIsolated (run-length < {args.patience}) morph events with >= {args.patience} "
              f"follow-up chunks available: {isolated_morph_count}")
        print(f"  of those, next {args.patience} chunks all clean (\"recovered\"): "
              f"{recovered_after_isolated} ({100*recovered_after_isolated/isolated_morph_count:.1f}%)")

    (out_dir / "episode_results.json").write_text(json.dumps(all_episode_results, indent=2))
    (out_dir / "aggregate.json").write_text(json.dumps(dict(
        agg=agg, n_episodes=n_eps, n_immediate_terminate=n_immediate_term,
        n_patience_terminate=n_patience_term, patience=args.patience,
        morph_run_length_hist=np.bincount(np.array(morph_run_lengths, dtype=int)).tolist() if morph_run_lengths else [],
        isolated_morph_count=isolated_morph_count, recovered_after_isolated=recovered_after_isolated,
    ), indent=2))
    print(f"\nFull per-episode results -> {out_dir/'episode_results.json'}")
    print(f"Aggregate summary -> {out_dir/'aggregate.json'}")

    # -------------------------------------------------- visual spot-check samples
    import cv2
    rng = np.random.default_rng(0)

    def save_event_strip(events, tag, n_save):
        if not events:
            print(f"No {tag} events to sample.")
            return
        idx = rng.choice(len(events), size=min(n_save, len(events)), replace=False)
        for i in idx:
            zpath, ep, chunk, end_frame = events[i][0], events[i][1], events[i][2], events[i][3]
            z = zarr.open(zpath, mode="r")
            ends = z["meta/episode_ends"][:]
            starts = np.concatenate([[0], ends[:-1]])
            s = int(starts[ep])
            lo = max(0, end_frame - 2)
            hi = min(int(ends[ep] - s), end_frame + 3)
            frames = z["data/img"][s + lo: s + hi]
            tiles = []
            for j, f in enumerate(frames):
                im = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                im = cv2.resize(im, (160, 160), interpolation=cv2.INTER_NEAREST)
                cv2.putText(im, f"f{lo+j}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                tiles.append(im)
            strip = np.concatenate(tiles, axis=1)
            zname = Path(zpath).stem
            label = f"{zname} ep{ep} chunk{chunk} end{end_frame}"
            cv2.putText(strip, label, (2, strip.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imwrite(str(out_dir / f"{tag}_{zname}_ep{ep:03d}_c{chunk:03d}.png"), strip)
        print(f"Saved {len(idx)} {tag} spot-check strips to {out_dir}/")

    save_event_strip(morph_events, "morph", args.n_visual_morph)
    save_event_strip(jump_events, "jump", args.n_visual_jump)


if __name__ == "__main__":
    main()

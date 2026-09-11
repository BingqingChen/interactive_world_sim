"""Build RLPD's offline demo_buffer from real rotate-T demo data
(datasets/scaling_v3/real/{A..H}.zarr, 800 episodes total), for condition 2
(WM-online + real-offline-data RLPD).

Per user direction: since the real zarr shards only store img/action/state
(no env_state/object pose -- confirmed earlier, and exact sim-replay would
need the original per-episode collection seed, which isn't recoverable from
the data alone), the offline reward uses the SAME validated pixel-angle
heuristic as the online WM env (_robust_angle_end, morph_terminate_patience=3,
reward=-0.01-on-morph, angle_prev-frozen-on-morph -- see
world_model_iws_rotate_t_env.py's chunk_step, kept in sync by hand) rather
than ground-truth eval_success. This was a real, deliberate simplification
of the original plan (which called for exact ground-truth reward on the
offline side specifically, accepting an online/offline reward-function
mismatch as a documented caveat) -- now BOTH sides use the identical reward
function, so that asymmetry caveat no longer applies. Real (non-hallucinated)
frames should if anything score better under this estimator than WM frames
do (confirmed via the full-pool validation: no reason to expect morph/jump
events on real footage beyond ordinary tracking noise).

Reuses render_imagined_pool_sample_videos.process_episode_full verbatim (same
reward/accept/morph logic, applied to real frames instead of imagined ones)
-- not reimplemented, to avoid the reward-logic-drift class of bug already
hit once this session (build_env() silently shadowing a tuned default).

R=200 per seed, disjoint slices over the combined 800-episode index (shards
A..H concatenated in that order): seed s -> global episodes
[(s-1)*200, s*200) -- the same disjoint-per-seed convention the BC scaling
sweep used (run_scaling_v3_sweep.py).

Packaging: one Trajectory per kept episode (T=that episode's own chunk count
up to its first termination/truncation, B=1), via
EmbodiedTrajectoryBuilder + ChunkStepResult, saved through
TrajectoryReplayBuffer(auto_save=True, trajectory_format="pt") -- the exact
pattern examples/embodiment/collect_real_data.py uses for its own
non-rollout-worker offline collection, confirmed via that file, not guessed.
No separate save_checkpoint() call needed: auto_save writes each trajectory's
.pt file + metadata.json + trajectory_index.json as add_trajectories() is
called; buffer.close() flushes the async save executor at the end.

Usage:
  MUJOCO_GL=egl /home/jacobhb/RLinf/.venv/bin/python \
      rlinf_integration/build_rlpd_offline_rotate_t.py \
      --seed 1 --state_history single \
      --out_dir outputs/rlpd_offline_demo_r200_seed1_single
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

WORKTREE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE_ROOT / "rlinf_integration"))
sys.path.insert(0, str(WORKTREE_ROOT / "scripts"))
sys.path.insert(0, str(WORKTREE_ROOT / "scripts" / "data_collection"))

from validate_reward_heuristics_full_pool import build_thresholded_env  # noqa: E402
from render_imagined_pool_sample_videos import process_episode_full  # noqa: E402

from rlinf.data.schema.embodied_trajectory_builder import EmbodiedTrajectoryBuilder  # noqa: E402
from rlinf.data.schema.embodied_types import ChunkStepResult  # noqa: E402
from rlinf.data.storage.replay import TrajectoryReplayBuffer  # noqa: E402

SHARD_LETTERS = "ABCDEFGH"
REAL_ROOT = WORKTREE_ROOT / "datasets" / "scaling_v3" / "real"


def build_combined_index():
    """(shard_letter, local_ep_idx) for global episode index 0..799, shards
    concatenated in A..H order -- matches the BC sweep's own convention."""
    index = []
    zarr_cache = {}
    for letter in SHARD_LETTERS:
        zpath = str(REAL_ROOT / f"{letter}.zarr")
        z = zarr.open(zpath, mode="r")
        zarr_cache[letter] = z
        n_ep = len(z["meta/episode_ends"][:])
        for local_ep in range(n_ep):
            index.append((letter, local_ep))
    return index, zarr_cache


def wrap_obs_single_frame(img_u8, state_row, state_history, prev_state_row=None):
    """Matches IWSRotateTSimEnv._wrap_obs's RLinf-native contract exactly:
    main_images (H,W,3) uint8 [0,255], states (state_dim,) f32 -- single
    current frame, no frame-history dim (image); states optionally
    concatenates the previous raw step's agent_pos too (concat_state)."""
    main_images = torch.from_numpy(img_u8.astype(np.float32))  # (H,W,3) [0,255]
    if state_history == "single":
        states = torch.from_numpy(state_row.astype(np.float32))
    else:
        assert prev_state_row is not None
        states = torch.from_numpy(
            np.concatenate([prev_state_row, state_row]).astype(np.float32))
    return {"main_images": main_images, "states": states}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--state_history", choices=["single", "concat_state"], default="single")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or str(
        WORKTREE_ROOT / "outputs" / f"rlpd_offline_demo_r{args.R}_seed{args.seed}_{args.state_history}")

    env = build_thresholded_env()
    n_act = env.n_act
    patience = env.morph_terminate_patience
    print(f"n_act={n_act} morph_terminate_patience={patience} state_history={args.state_history}")

    combined_index, zarr_cache = build_combined_index()
    assert len(combined_index) == 800, f"expected 800 real episodes, got {len(combined_index)}"
    lo = (args.seed - 1) * args.R
    hi = lo + args.R
    picks = combined_index[lo:hi]
    print(f"seed={args.seed}: global episodes [{lo}, {hi}) -> {len(picks)} episodes")

    buffer = TrajectoryReplayBuffer(
        seed=args.seed, enable_cache=False, auto_save=True,
        auto_save_path=out_dir, trajectory_format="pt")

    n_kept, n_skipped, n_success, n_morph_term, n_natural_end = 0, 0, 0, 0, 0
    total_chunks = 0
    for i, (letter, local_ep) in enumerate(picks):
        z = zarr_cache[letter]
        ends = z["meta/episode_ends"][:]
        starts = np.concatenate([[0], ends[:-1]])
        s, e = int(starts[local_ep]), int(ends[local_ep])
        imgs = z["data/img"][s:e]
        actions_raw = z["data/action"][s:e]  # (L,4)
        states_raw = z["data/state"][s:e]    # (L,4) agent_pos
        L = len(imgs)
        if L < n_act:
            n_skipped += 1
            continue

        per_chunk, f0_area = process_episode_full(env, imgs, n_act, patience)
        if not per_chunk:
            n_skipped += 1
            continue

        rollout = EmbodiedTrajectoryBuilder(max_episode_length=len(per_chunk))
        prev_frame_idx = 0
        kept = 0
        ended_reason = "natural_end"
        for c in per_chunk:
            end = c["end_frame"]
            is_last_natural = (end == per_chunk[-1]["end_frame"])
            term = c["success"]
            trunc = c["morph_terminate"] or (is_last_natural and not term)

            action_tensor = torch.from_numpy(
                actions_raw[end - n_act:end].reshape(-1).astype(np.float32)
            ).unsqueeze(0)  # (1, 4*n_act) -- matches actor.model.action_dim=32
            reward_tensor = torch.tensor([[c["reward"]]], dtype=torch.float32)
            term_tensor = torch.tensor([[term]], dtype=torch.bool)
            trunc_tensor = torch.tensor([[trunc]], dtype=torch.bool)
            done_tensor = term_tensor | trunc_tensor

            step_result = ChunkStepResult(
                actions=action_tensor, rewards=reward_tensor, dones=done_tensor,
                terminations=term_tensor, truncations=trunc_tensor,
                forward_inputs={"action": action_tensor},
            )
            rollout.append_step_result(step_result)

            curr_frame_idx = prev_frame_idx
            curr_prev_state_idx = max(0, curr_frame_idx - 1)
            next_frame_idx = end - 1
            next_prev_state_idx = max(0, next_frame_idx - 1)
            curr_obs = wrap_obs_single_frame(
                imgs[curr_frame_idx], states_raw[curr_frame_idx], args.state_history,
                states_raw[curr_prev_state_idx] if args.state_history == "concat_state" else None)
            next_obs = wrap_obs_single_frame(
                imgs[next_frame_idx], states_raw[next_frame_idx], args.state_history,
                states_raw[next_prev_state_idx] if args.state_history == "concat_state" else None)
            # Unsqueeze to (1, ...) -- EmbodiedTrajectoryBuilder expects per-step
            # obs dicts shaped (B, ...), B=1 here (one real episode per Trajectory).
            curr_obs = {k: v.unsqueeze(0) for k, v in curr_obs.items()}
            next_obs = {k: v.unsqueeze(0) for k, v in next_obs.items()}
            rollout.append_transitions(curr_obs=curr_obs, next_obs=next_obs)

            prev_frame_idx = next_frame_idx
            kept += 1
            if term:
                ended_reason = "success"
            elif c["morph_terminate"]:
                ended_reason = "morph_terminate"
            if term or trunc:
                break

        if kept == 0:
            n_skipped += 1
            continue

        trajectory = rollout.to_trajectory()
        buffer.add_trajectories([trajectory])
        n_kept += 1
        total_chunks += kept
        if ended_reason == "success":
            n_success += 1
        elif ended_reason == "morph_terminate":
            n_morph_term += 1
        else:
            n_natural_end += 1
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(picks)}] kept={n_kept} skipped={n_skipped}")

    buffer.close()
    print(f"\nDone. {n_kept} episodes -> {out_dir}")
    print(f"  skipped (degenerate/too short): {n_skipped}")
    print(f"  ended in success: {n_success}  morph_terminate: {n_morph_term}  natural/timeout end: {n_natural_end}")
    print(f"  total chunks (transitions) written: {total_chunks}")


if __name__ == "__main__":
    main()

"""Build RLPD's offline demo_buffer from real rotate-T demo data
(datasets/scaling_v3/real/{A..H}.zarr, 800 episodes), R episodes per seed on
disjoint slices [(s-1)*R, s*R) of the A..H-concatenated index -- the BC scaling
sweep's convention.

Reward: the same validated pixel-angle heuristic as the online WM env
(render_imagined_pool_sample_videos.process_episode_full, reused verbatim). The
zarrs store no object pose and the original collection seeds aren't
recoverable, so ground-truth sim replay isn't possible. In a *real-sim* RLPD run
the online reward is ground truth, so online and offline rewards differ there.

Actions must match what the online env consumes:
  --action_mode absolute: flattened (n_act*4) EE-xy targets.
  --action_mode delta: normalized per-step deltas, a_k = clip((target_k -
    prev) / max_step, -1, 1), where prev starts at the reset EE-xy (state[0])
    and is re-integrated exactly as real_sim_chunk_server.py does (clip into
    the demo box), so reconstruction error can't accumulate.
Obs: a single current frame, {"main_images": (H,W,3) [0,255], "states"}, with
states empty for --state_history none.

Packaging follows examples/embodiment/collect_real_data.py: one Trajectory per
episode (B=1, T = chunks up to the first termination/truncation) through
EmbodiedTrajectoryBuilder + ChunkStepResult into
TrajectoryReplayBuffer(auto_save=True, trajectory_format="pt").

Usage:
  MUJOCO_GL=egl ~/RLinf/.venv/bin/python rlinf_integration/build_rlpd_offline_rotate_t.py \
      --seed 1 --state_history none --action_mode delta
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
from real_sim_chunk_server import DEMO_TARGET_LO, DEMO_TARGET_HI  # noqa: E402

from rlinf.data.schema.embodied_trajectory_builder import EmbodiedTrajectoryBuilder  # noqa: E402
from rlinf.data.schema.embodied_types import ChunkStepResult  # noqa: E402
from rlinf.data.storage.replay import TrajectoryReplayBuffer  # noqa: E402

SHARD_LETTERS = "ABCDEFGH"
REAL_ROOT = WORKTREE_ROOT / "datasets" / "scaling_v3" / "real"


def build_combined_index():
    index, zarr_cache = [], {}
    for letter in SHARD_LETTERS:
        z = zarr.open(str(REAL_ROOT / f"{letter}.zarr"), mode="r")
        zarr_cache[letter] = z
        index += [(letter, ep) for ep in range(len(z["meta/episode_ends"][:]))]
    return index, zarr_cache


def wrap_obs(img_u8, states_raw, idx, state_history):
    obs = {"main_images": torch.from_numpy(img_u8.astype(np.float32))}
    if state_history == "single":
        obs["states"] = torch.from_numpy(states_raw[idx].astype(np.float32))
    elif state_history == "concat_state":
        obs["states"] = torch.from_numpy(
            np.concatenate([states_raw[max(0, idx - 1)], states_raw[idx]]).astype(np.float32))
    else:
        obs["states"] = torch.zeros(1, dtype=torch.float32)  # image-only: constant zero state
    return {k: v.unsqueeze(0) for k, v in obs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--state_history", choices=["none", "single", "concat_state"], default="none")
    ap.add_argument("--action_mode", choices=["absolute", "delta"], default="delta")
    ap.add_argument("--max_step", type=float, default=0.04)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or str(
        WORKTREE_ROOT / "outputs" /
        f"rlpd_offline_demo_r{args.R}_seed{args.seed}_{args.state_history}_{args.action_mode}")
    lo, hi = np.array(DEMO_TARGET_LO), np.array(DEMO_TARGET_HI)

    env = build_thresholded_env()
    n_act, patience = env.n_act, env.morph_terminate_patience
    print(f"n_act={n_act} patience={patience} state_history={args.state_history} "
          f"action_mode={args.action_mode} max_step={args.max_step}")

    combined_index, zarr_cache = build_combined_index()
    assert len(combined_index) == 800, len(combined_index)
    lo_ep = (args.seed - 1) * args.R
    picks = combined_index[lo_ep:lo_ep + args.R]
    print(f"seed={args.seed}: global episodes [{lo_ep}, {lo_ep + args.R})")

    buffer = TrajectoryReplayBuffer(seed=args.seed, enable_cache=False, auto_save=True,
                                    auto_save_path=out_dir, trajectory_format="pt")

    n_kept = n_skipped = n_success = n_morph_term = n_natural_end = total_chunks = 0
    clipped = total_steps = 0
    max_recon_err = 0.0
    for i, (letter, local_ep) in enumerate(picks):
        z = zarr_cache[letter]
        ends = z["meta/episode_ends"][:]
        starts = np.concatenate([[0], ends[:-1]])
        s, e = int(starts[local_ep]), int(ends[local_ep])
        imgs = z["data/img"][s:e]
        actions_raw = z["data/action"][s:e].astype(np.float64)
        states_raw = z["data/state"][s:e]
        L = len(imgs)
        if L < n_act:
            n_skipped += 1
            continue
        per_chunk, _ = process_episode_full(env, imgs, n_act, patience)
        if not per_chunk:
            n_skipped += 1
            continue

        rollout = EmbodiedTrajectoryBuilder(max_episode_length=len(per_chunk))
        prev_target = states_raw[0].astype(np.float64)
        prev_frame_idx = 0
        kept = 0
        ended_reason = "natural_end"
        for c in per_chunk:
            end = c["end_frame"]
            term = c["success"]
            trunc = c["morph_terminate"] or (end == per_chunk[-1]["end_frame"] and not term)

            chunk_abs = actions_raw[end - n_act:end]
            if args.action_mode == "delta":
                chunk_act = np.empty_like(chunk_abs)
                for k in range(n_act):
                    raw = (chunk_abs[k] - prev_target) / args.max_step
                    clipped += int((np.abs(raw) > 1.0).sum())
                    chunk_act[k] = np.clip(raw, -1.0, 1.0)
                    prev_target = np.clip(prev_target + chunk_act[k] * args.max_step, lo, hi)
                    max_recon_err = max(max_recon_err, float(np.abs(prev_target - chunk_abs[k]).max()))
                total_steps += chunk_abs.size
            else:
                chunk_act = chunk_abs
            action_tensor = torch.from_numpy(chunk_act.reshape(-1).astype(np.float32)).unsqueeze(0)

            done_tensor = torch.tensor([[term or trunc]], dtype=torch.bool)
            rollout.append_step_result(ChunkStepResult(
                actions=action_tensor,
                rewards=torch.tensor([[c["reward"]]], dtype=torch.float32),
                dones=done_tensor,
                terminations=torch.tensor([[term]], dtype=torch.bool),
                truncations=torch.tensor([[trunc]], dtype=torch.bool),
                forward_inputs={"action": action_tensor},
            ))
            # action[k] produces frame k+1, so a chunk ending at `end` leaves frame `end`.
            next_frame_idx = min(end, L - 1)
            rollout.append_transitions(
                curr_obs=wrap_obs(imgs[prev_frame_idx], states_raw, prev_frame_idx, args.state_history),
                next_obs=wrap_obs(imgs[next_frame_idx], states_raw, next_frame_idx, args.state_history))
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
        buffer.add_trajectories([rollout.to_trajectory()])
        n_kept += 1
        total_chunks += kept
        n_success += ended_reason == "success"
        n_morph_term += ended_reason == "morph_terminate"
        n_natural_end += ended_reason == "natural_end"
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(picks)}] kept={n_kept} skipped={n_skipped}")

    buffer.close()
    print(f"\nDone. {n_kept} episodes -> {out_dir}")
    print(f"  skipped: {n_skipped}  success: {n_success}  morph_terminate: {n_morph_term}  "
          f"natural/timeout end: {n_natural_end}  transitions: {total_chunks}")
    if args.action_mode == "delta":
        print(f"  delta clipping: {clipped}/{total_steps} action values ({100*clipped/max(total_steps,1):.3f}%), "
              f"max |reconstructed - demo target| = {max_recon_err:.4f} m")


if __name__ == "__main__":
    main()

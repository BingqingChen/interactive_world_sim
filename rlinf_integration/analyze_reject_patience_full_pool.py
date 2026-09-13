"""How often would early stopping end imagined episodes if EVERY reject type counted
toward patience, not just morph_detected?

world_model_iws_rotate_t_env.py truncates an episode after morph_terminate_patience (3)
consecutive morph_detected chunks (T mask area > 1.20x frame 0). The other rejects never
end an episode: lost-track and low-IoU shape rejects carry the angle forward, and after
stuck_reject_limit (3) consecutive rejects a jump candidate is force-accepted. The agreed
spec was "if either reject criterion persists for patience=3, terminate". This measures
what that costs on the 800-episode imagined pool before the env is changed.

Runs the production IWSRotateTWorldEnv._robust_angle_end on every chunk (thresholds from
the real _build_dataset, via validate_reward_heuristics_full_pool.build_thresholded_env),
with two settings that mirror the proposed rule rather than the current one:
  * stuck-prev recovery disabled: under the proposed rule the third consecutive reject
    terminates instead of force-accepting;
  * angle_prev frozen on morph_detected chunks, as chunk_step does.
Each chunk is labelled by what that call itself logged: ok / lost / shape (IoU below
floor) / jump, plus the morph_detected flag.

Usage (from the worktree root):
  /home/jacobhb/RLinf/.venv/bin/python rlinf_integration/analyze_reject_patience_full_pool.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_reward_heuristics_full_pool import build_thresholded_env, run_length_stats  # noqa: E402
from world_model_iws_rotate_t_env import REWARD_THETAS  # noqa: E402
from collect_imagined_rotate_t import make_templates, red_mask  # noqa: E402

CODE = {"ok": ".", "lost": "L", "shape": "S", "jump": "J"}
VARIANTS = {
    "morph_only": lambda kind, morph: morph,
    "morph_or_jump": lambda kind, morph: morph or kind == "jump",
    "any_reject": lambda kind, morph: morph or kind != "ok",
}


def variant_flags(name, chunks, shrink):
    """Per-chunk bad flags for a rule. "any_reject_or_shrink" is the rule adopted for the
    world-model env on 2026-09-13: any_reject plus area shrink below area_ratio_low in
    any of the chunk's last 3 frames."""
    if name == "any_reject_or_shrink":
        return [morph or kind != "ok" or s for (kind, morph, _), s in zip(chunks, shrink)]
    return [VARIANTS[name](kind, morph) for kind, morph, _ in chunks]


def label_episode(env, imgs, n_act, shrink_out=None):
    templates, tc = make_templates(imgs[0], thetas=REWARD_THETAS)
    env._templates = [templates]
    env._tc = [tc]
    env._frame0_area = [int(red_mask(imgs[0]).sum())]
    env.angle_prev = np.zeros(1, dtype=np.float64)
    env._consec_reject = np.zeros(1, dtype=np.int64)
    env._jump_reject_count = 0
    env._lost_track_count = 0
    env._morph_reject_count = 0
    env._stuck_recovery_count = 0
    env._debug_log = []

    k = min(3, n_act)
    chunks = []
    for c in range(1, len(imgs) // n_act + 1):
        end = c * n_act
        n_log = len(env._debug_log)
        angle_end, accepted, morph, shrink = env._robust_angle_end(
            [imgs[j] for j in range(end - k, end)], 0
        )
        if shrink_out is not None:
            shrink_out.append(bool(shrink))
        new_kinds = [d["kind"] for d in env._debug_log[n_log:]]
        assert "stuck_recovery" not in new_kinds
        if accepted:
            kind = "ok"
        elif "jump" in new_kinds:
            kind = "jump"
        elif "morph" in new_kinds:  # the reject path's name for a low-IoU shape reject
            kind = "shape"
        else:
            kind = "lost"
        if not morph:
            env.angle_prev[0] = angle_end
        chunks.append((kind, bool(morph), float(np.degrees(angle_end))))
    return chunks


def trigger_index(flags, patience):
    """0-based chunk index at which a run of `patience` bad chunks completes, or None."""
    for start, length in run_length_stats(flags):
        if length >= patience:
            return start + patience - 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarrs", nargs="+", default=[
        "datasets/scaling_v3/imag/half1.zarr", "datasets/scaling_v3/imag/half2.zarr"])
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--out_dir", default="outputs/reward_reject_patience_full_pool")
    ap.add_argument("--max_episodes", type=int, default=None, help="per zarr, for a smoke test")
    args = ap.parse_args()

    env = build_thresholded_env()
    env.stuck_reject_limit = 10**9
    n_act = env.n_act
    patience = args.patience
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    for zpath in args.zarrs:
        z = zarr.open(zpath, mode="r")
        ends = z["meta/episode_ends"][:]
        starts = np.concatenate([[0], ends[:-1]])
        n_eps = len(ends) if args.max_episodes is None else min(len(ends), args.max_episodes)
        for ep in range(n_eps):
            imgs = z["data/img"][int(starts[ep]):int(ends[ep])]
            if len(imgs) < n_act:
                continue
            shrink = []
            chunks = label_episode(env, imgs, n_act, shrink_out=shrink)
            episodes.append(dict(zarr=zpath, episode=ep, chunks=chunks, shrink=shrink))
            if len(episodes) % 50 == 0:
                print(f"{len(episodes)} episodes labelled", flush=True)

    total_chunks = sum(len(e["chunks"]) for e in episodes)
    kind_counts = Counter(kind for e in episodes for kind, _, _ in e["chunks"])
    morph_count = sum(m for e in episodes for _, m, _ in e["chunks"])
    summary = dict(
        patience=patience, n_episodes=len(episodes), total_chunks=total_chunks,
        chunk_kinds=dict(kind_counts), morph_detected_chunks=int(morph_count),
        shrink_detected_chunks=int(sum(s for e in episodes for s in e["shrink"])),
        thresholds=dict(jump_reject_deg=env.jump_reject_deg, min_iou=env.min_iou,
                        area_ratio=[env.area_ratio_low, env.area_ratio_high],
                        terminal_deg=env.terminal_deg),
        variants={},
    )

    for name in [*VARIANTS, "any_reject_or_shrink"]:
        trig_chunks, removed, run_lengths = [], 0, []
        isolated, recovered = 0, 0
        success_eps, cut_before_success = 0, 0
        trigger_patterns = Counter()
        for e in episodes:
            flags = variant_flags(name, e["chunks"], e["shrink"])
            runs = run_length_stats(flags)
            run_lengths.extend(length for _, length in runs)
            for start, length in runs:
                if length < patience and start + length + patience <= len(flags):
                    isolated += 1
                    recovered += not any(flags[start + length:start + length + patience])
            t = trigger_index(flags, patience)
            succ = next((i for i, (kind, morph, deg) in enumerate(e["chunks"])
                         if kind == "ok" and not morph and deg <= env.terminal_deg), None)
            if succ is not None:
                success_eps += 1
                cut_before_success += t is not None and t < succ
            if t is not None:
                trig_chunks.append(t + 1)
                removed += len(flags) - (t + 1)
                trigger_patterns["".join(
                    CODE[kind] + ("m" if morph else "")
                    for kind, morph, _ in e["chunks"][t - patience + 1:t + 1])] += 1
        hist = Counter(run_lengths)
        summary["variants"][name] = dict(
            n_truncated=len(trig_chunks),
            pct_truncated=round(100 * len(trig_chunks) / max(len(episodes), 1), 1),
            trigger_chunk_median=float(np.median(trig_chunks)) if trig_chunks else None,
            trigger_chunk_p10_p90=[float(np.percentile(trig_chunks, 10)),
                                   float(np.percentile(trig_chunks, 90))] if trig_chunks else None,
            pct_chunks_removed=round(100 * removed / max(total_chunks, 1), 1),
            bad_run_length_hist={int(k): int(v) for k, v in sorted(hist.items())},
            isolated_runs=isolated,
            pct_isolated_recovered=round(100 * recovered / max(isolated, 1), 1),
            proxy_success_episodes=success_eps,
            truncated_before_proxy_success=int(cut_before_success),
            top_trigger_patterns=trigger_patterns.most_common(12),
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "episode_labels.json").write_text(json.dumps([
        dict(zarr=e["zarr"], episode=e["episode"],
             labels=[CODE[kind] + ("m" if morph else "") + ("s" if s else "")
                     for (kind, morph, _), s in zip(e["chunks"], e["shrink"])],
             angle_deg=[round(deg, 1) for _, _, deg in e["chunks"]])
        for e in episodes
    ]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

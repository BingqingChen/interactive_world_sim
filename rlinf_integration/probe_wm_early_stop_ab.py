"""A/B check of the WM env's chunk readout after the per-row-reset / any-reject rewrite.

WM Run A (redesigned env) finishes ~4 episodes per 16 chunk-steps, i.e. episodes of ~4
chunks: early stop fires almost immediately. This separates "the new code is broken"
from "early RL actions make the WM hallucinate at once" by running the previous env
(5a1a7f3, batch reset, growth-only early stop) and the current one from the same real
reset seeds, under two action regimes:
  smooth_walk  absolute EE-xy random walk (the regime where the old env accepted ~98%)
  delta_random normalized deltas tanh(N(0,1)), like an untrained SAC actor
Per chunk: angle read accepted or reject kind, area growth (morph_detected) and area
shrink (recomputed from the frames for both versions, since the old env has no shrink
flag), by chunk index since the row's reset.

Usage (from the worktree root; the previous env file must sit in rlinf_integration/):
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 ~/RLinf/.venv/bin/python \
      rlinf_integration/probe_wm_early_stop_ab.py --old_env rlinf_integration/_probe_old_env_5a1a7f3.py
"""
import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WT = HERE.parent

REAL_STEP_DELTA_STD = np.array([0.00443962, 0.00653751, 0.00439547, 0.00649021], np.float32)


def load_env_class(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.IWSRotateTWorldEnv, mod


def build_env(cls, B, device, action_mode, seed_base, old):
    cfg = OmegaConf.create({
        "n_action_steps": 8, "wm_max_chunks": 40, "dec_infer_steps": 2,
        "x_range": [-0.06, 0.06], "y_range": [-0.06, 0.06], "env_seed_base": seed_base,
        "state_history": "none", "action_mode": action_mode, "max_step": 0.04,
        "early_stop_rule": "any_reject", "morph_terminate_patience": 3,
        "stuck_reject_limit": 3 if old else 0,
        "wm_ckpt": "/home/jacobhb/projects/worth_doing/interactive_world_sim/ckpts/push_t/epoch=3-step=90000.ckpt",
        "wm_config": "pusht_mujoco",
        "feasible_mask": "/home/jacobhb/projects/worth_doing/interactive_world_sim/datasets/feasible_mask_v3.json",
    })
    env = object.__new__(cls)
    env.cfg = cfg
    env.device = torch.device(device)
    env.num_envs = B
    env.record_metrics = True
    env.auto_reset = True
    env.ignore_terminations = False
    env.use_rel_reward = False
    env._is_start = True
    env._elapsed_steps = 0
    env.video_cfg = None
    env.prev_step_reward = torch.zeros(B, dtype=torch.float32, device=env.device)
    env.enable_kir = True
    env.dataset = env._build_dataset(cfg)
    env._init_metrics()
    return env


def run_variant(label, cls, mod, old, action_regime, args):
    action_mode = "absolute" if action_regime == "smooth_walk" else "delta"
    env = build_env(cls, args.B, args.device, action_mode, args.seed_base, old)
    B = env.num_envs
    env.reset()
    records, frames = [], {}
    orig = env._robust_angle_end
    counters = ("_jump_reject_count", "_lost_track_count", "_morph_reject_count")

    def wrapped(dec_u8_last3, row):
        before = [getattr(env, c) for c in counters]
        out = orig(dec_u8_last3, row)
        after = [getattr(env, c) for c in counters]
        kind = "ok" if out[1] else {0: "jump", 1: "lost", 2: "shape"}.get(
            int(np.argmax(np.array(after) - np.array(before))), "?")
        f0 = env._frame0_area[row]
        shrink = False
        for fr in dec_u8_last3:
            _, _, area = mod.est_angle_with_conf(fr, env._templates[row], env._tc[row],
                                                 thetas=mod.REWARD_THETAS)
            shrink |= area < env.area_ratio_low * f0
        chunk = int(np.atleast_1d(env.steps)[row] if np.ndim(env.steps) else env.steps)
        records.append(dict(row=row, chunk=chunk, kind=kind, morph=bool(out[2]), shrink=bool(shrink)))
        if row < 4 and chunk <= 10:
            frames[(row, chunk)] = (dec_u8_last3[-1].copy(), kind, bool(out[2]), bool(shrink))
        return out

    env._robust_angle_end = wrapped
    rng = np.random.default_rng(args.action_seed)
    walk = env.curr_state.detach().cpu().numpy().copy()
    stats = env.wm.normalizer["action"].get_input_stats()
    a_min = stats["min"].detach().cpu().numpy().reshape(-1)
    a_max = stats["max"].detach().cpu().numpy().reshape(-1)
    lengths, early_stops, successes = [], 0, 0
    for _ in range(args.n_chunks):
        steps_before = np.atleast_1d(env.steps).copy()
        if action_regime == "smooth_walk":
            prev = walk.copy()
            walk = np.clip(prev + rng.normal(0.0, REAL_STEP_DELTA_STD * np.sqrt(8), (B, 4)), a_min, a_max)
            alphas = np.linspace(1.0 / 8, 1.0, 8, dtype=np.float32)
            actions = (prev[:, None, :] + alphas[None, :, None] * (walk - prev)[:, None, :]).astype(np.float32)
        else:
            actions = np.tanh(rng.normal(0.0, 1.0, (B, 8, 4))).astype(np.float32)
        _, _, terms, truncs, infos = env.chunk_step(torch.from_numpy(actions.reshape(B, 1, 32)))
        done = (terms | truncs).reshape(-1).cpu().numpy()
        successes += int(terms.reshape(-1).cpu().numpy().sum())
        early = infos[0].get("final_info", {}).get("episode", {}).get("early_stop")
        if early is not None and done.any():
            early_stops += int(early.reshape(-1).cpu().numpy()[done].sum())
        for i in np.flatnonzero(done):
            lengths.append(int(steps_before[i if steps_before.size > 1 else 0]) + 1)
        cur = env.curr_state.detach().cpu().numpy()
        for i in np.flatnonzero(done):
            walk[i] = cur[i]
    if hasattr(env, "close"):
        env.close()

    n = len(records)
    bad = [r for r in records if r["kind"] != "ok" or r["morph"] or r["shrink"]]
    by_bucket = {}
    for lo, hi in ((1, 3), (4, 8), (9, 40)):
        sel = [r for r in records if lo <= r["chunk"] <= hi]
        nb = sum(1 for r in sel if r["kind"] != "ok" or r["morph"] or r["shrink"])
        by_bucket[f"chunks {lo}-{hi}"] = f"{nb}/{len(sel)} bad"
    summary = dict(
        variant=label, reads=n,
        accepted=round(sum(r["kind"] == "ok" for r in records) / max(n, 1), 3),
        bad=round(len(bad) / max(n, 1), 3),
        kinds=dict(Counter(r["kind"] for r in records)),
        morph=sum(r["morph"] for r in records), shrink=sum(r["shrink"] for r in records),
        by_chunk_since_reset=by_bucket,
        episodes_finished=len(lengths), mean_episode_chunks=round(float(np.mean(lengths)), 2) if lengths else None,
        early_stops=early_stops, successes=successes,
    )
    print(json.dumps(summary), flush=True)
    return summary, frames


def contact_sheet(frames, path):
    rows = []
    for row in range(4):
        tiles = []
        for chunk in range(1, 11):
            if (row, chunk) not in frames:
                tiles.append(np.zeros((192, 192, 3), np.uint8))
                continue
            img, kind, morph, shrink = frames[(row, chunk)]
            t = cv2.resize(img, (192, 192), interpolation=cv2.INTER_NEAREST)
            txt = f"r{row} c{chunk} {kind}{' M' if morph else ''}{' S' if shrink else ''}"
            cv2.rectangle(t, (0, 0), (192, 18), (0, 0, 0), -1)
            cv2.putText(t, txt, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        rows.append(np.concatenate(tiles, axis=1))
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_env", required=True)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--n_chunks", type=int, default=12)
    ap.add_argument("--seed_base", type=int, default=860_000)
    ap.add_argument("--action_seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default=str(WT / "outputs/probe_wm_early_stop_ab"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    new_cls, new_mod = load_env_class(HERE / "world_model_iws_rotate_t_env.py", "wm_env_new")
    old_cls, old_mod = load_env_class(Path(args.old_env).resolve(), "wm_env_old")
    summaries = []
    for regime in ("smooth_walk", "delta_random"):
        for label, cls, mod, old in (("old", old_cls, old_mod, True), ("new", new_cls, new_mod, False)):
            s, frames = run_variant(f"{label}/{regime}", cls, mod, old, regime, args)
            summaries.append(s)
            contact_sheet(frames, out_dir / f"contact_{label}_{regime}.png")
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    main()

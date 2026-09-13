"""Render every imagined-pool episode that early stopping would cut under the broadest
candidate rule, for visual diagnosis of what each rule actually terminates on.

Candidate rules (patience = 3 consecutive "bad" chunks, one shared counter):
  (a) morph_only    -- bad = morph_detected (T mask area > 1.20x frame 0); current env
  (b) morph_or_jump -- bad = morph_detected or jump reject
  (c) any_reject    -- bad = morph_detected or any reject (jump / lost track / shape)
The rules are nested, so every episode cut by (a) or (b) is also cut by (c); this renders
all (c) cuts, filed by the EARLIEST rule that cuts the episode:
  a_morph/        cut by (a) already
  b_jump/         first cut by (b): a jump reject is needed to complete the run
  c_lost_shape/   cut only by (c): lost-track / low-IoU shape rejects complete the run

Chunk labels come from analyze_reject_patience_full_pool.label_episode (production
_robust_angle_end, stuck-prev recovery disabled, angle_prev frozen on morph), so they are
exactly the labels the summary statistics were computed from. The overlay also shows the
last frame's best-match IoU and area ratio, recomputed read-only per chunk.

Usage (from the worktree root):
  /home/jacobhb/RLinf/.venv/bin/python rlinf_integration/render_reject_patience_trigger_videos.py
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_reject_patience_full_pool import VARIANTS, CODE, label_episode, trigger_index  # noqa: E402
from validate_reward_heuristics_full_pool import build_thresholded_env  # noqa: E402
from render_imagined_pool_sample_videos import label  # noqa: E402
from world_model_iws_rotate_t_env import REWARD_THETAS  # noqa: E402
from collect_imagined_rotate_t import make_templates, est_angle_with_conf, red_mask  # noqa: E402

RULE_LETTER = {"morph_only": "a", "morph_or_jump": "b", "any_reject": "c"}
FOLDER = {"morph_only": "a_morph", "morph_or_jump": "b_jump", "any_reject": "c_lost_shape"}
KIND_TEXT = {"ok": "accepted", "lost": "REJECT lost-track", "shape": "REJECT shape (IoU)",
             "jump": "REJECT jump"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarrs", nargs="+", default=[
        "datasets/scaling_v3/imag/half1.zarr", "datasets/scaling_v3/imag/half2.zarr"])
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--out_dir", default="outputs/reject_patience_trigger_videos")
    args = ap.parse_args()

    env = build_thresholded_env()
    env.stuck_reject_limit = 10**9
    n_act = env.n_act
    patience = args.patience
    out_dir = Path(args.out_dir)
    for folder in FOLDER.values():
        (out_dir / folder).mkdir(parents=True, exist_ok=True)

    index = []
    for zpath in args.zarrs:
        z = zarr.open(zpath, mode="r")
        ends = z["meta/episode_ends"][:]
        starts = np.concatenate([[0], ends[:-1]])
        for ep in range(len(ends)):
            imgs = z["data/img"][int(starts[ep]):int(ends[ep])]
            if len(imgs) < n_act:
                continue
            chunks = label_episode(env, imgs, n_act)
            flags = {name: [bad(kind, morph) for kind, morph, _ in chunks]
                     for name, bad in VARIANTS.items()}
            trig = {name: trigger_index(f, patience) for name, f in flags.items()}
            if trig["any_reject"] is None:
                continue
            first_rule = next(name for name in VARIANTS if trig[name] is not None)
            succ = next((i for i, (kind, morph, deg) in enumerate(chunks)
                         if kind == "ok" and not morph and deg <= env.terminal_deg), None)
            cut_before_success = succ is not None and trig["any_reject"] < succ

            templates, tc = make_templates(imgs[0], thetas=REWARD_THETAS)
            f0_area = int(red_mask(imgs[0]).sum())
            iou_area = []
            for c in range(len(chunks)):
                _, iou, area = est_angle_with_conf(
                    imgs[(c + 1) * n_act - 1], templates, tc, thetas=REWARD_THETAS)
                iou_area.append((float(iou), area / f0_area if f0_area else 0.0))

            run_b = run_c = 0
            counts = []
            for c in range(len(chunks)):
                run_b = run_b + 1 if flags["morph_or_jump"][c] else 0
                run_c = run_c + 1 if flags["any_reject"][c] else 0
                counts.append((run_b, run_c))

            t = trig["any_reject"]
            pattern = "".join(CODE[k] + ("m" if m else "")
                              for k, m, _ in chunks[t - patience + 1:t + 1])
            tag = f"{Path(zpath).stem}_ep{ep:03d}"
            name = f"{tag}_cut{t + 1:02d}_{pattern}{'_CUT_BEFORE_SUCCESS' if cut_before_success else ''}.mp4"
            out_path = out_dir / FOLDER[first_rule] / name

            vid = []
            for frame_t in range(len(chunks) * n_act):
                c = frame_t // n_act
                kind, morph, deg = chunks[c]
                iou, area_ratio = iou_area[c]
                bad_c = flags["any_reject"][c]
                color = (255, 60, 60) if bad_c else (60, 255, 60)
                cut_here = [RULE_LETTER[r] for r in VARIANTS if trig[r] == c]
                cut_past = [RULE_LETTER[r] for r in VARIANTS if trig[r] is not None and trig[r] < c]
                lines = [
                    f"{tag}  frame {frame_t}  chunk {c + 1}/{len(chunks)}",
                    f"{KIND_TEXT[kind]}{'  +MORPH(area)' if morph else ''}",
                    f"angle={deg:+.1f}deg  iou={iou:.2f}  area={area_ratio:.2f}",
                    f"bad-run: (b) {counts[c][0]}/{patience}  (c) {counts[c][1]}/{patience}",
                ]
                if cut_here:
                    lines.append(">>> CUT HERE under " + ",".join(cut_here) + " <<<")
                elif cut_past:
                    lines.append("(already cut under " + ",".join(cut_past) + ")")
                if succ is not None and c == succ:
                    lines.append("PROXY SUCCESS (angle <= -80)")
                frame = cv2.resize(imgs[frame_t], (384, 384), interpolation=cv2.INTER_NEAREST)
                vid.append(label(frame, lines, color))
            imageio.mimwrite(out_path, np.stack(vid), fps=8, codec="libx264",
                             pixelformat="yuv420p", output_params=["-crf", "20"])

            index.append(dict(
                zarr=zpath, episode=ep, n_chunks=len(chunks), video=str(out_path),
                first_rule=RULE_LETTER[first_rule],
                cut_chunk={RULE_LETTER[r]: (trig[r] + 1 if trig[r] is not None else None)
                           for r in VARIANTS},
                trigger_pattern_c=pattern, proxy_success_chunk=(succ + 1 if succ is not None else None),
                cut_before_success_c=bool(cut_before_success),
                labels=[CODE[k] + ("m" if m else "") for k, m, _ in chunks],
            ))
            print(f"[{len(index)}] {out_path}", flush=True)

    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    by_rule = {letter: sum(1 for e in index if e["first_rule"] == letter) for letter in "abc"}
    print(f"rendered {len(index)} episodes; first cut by rule: {by_rule}")


if __name__ == "__main__":
    main()

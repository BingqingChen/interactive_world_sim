"""One contact sheet per early-stop rule folder (a_morph / b_jump / c_lost_shape) from
render_reject_patience_trigger_videos.py's index.json: for up to --per_sheet episodes
(evenly spaced through the folder), frame 0, the last frame of the chunk before the
3-chunk bad run that triggers rule (c), and the last frame of each of those 3 chunks.

Usage (from the worktree root):
  /home/jacobhb/RLinf/.venv/bin/python rlinf_integration/contact_sheets_reject_patience.py
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import zarr

FOLDER = {"a": "a_morph", "b": "b_jump", "c": "c_lost_shape"}


def tile(img, text, size=192):
    out = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(out, (0, 0), (size, 18), (0, 0, 0), -1)
    cv2.putText(out, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default="outputs/reject_patience_trigger_videos")
    ap.add_argument("--per_sheet", type=int, default=6)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--n_act", type=int, default=8)
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    index = json.loads((video_dir / "index.json").read_text())
    zarrs = {}
    for letter, folder in FOLDER.items():
        eps = [e for e in index if e["first_rule"] == letter]
        picks = [eps[i] for i in np.linspace(0, len(eps) - 1, min(args.per_sheet, len(eps))).round().astype(int)]
        rows = []
        for e in picks:
            z = zarrs.setdefault(e["zarr"], zarr.open(e["zarr"], mode="r"))
            ends = z["meta/episode_ends"][:]
            start = 0 if e["episode"] == 0 else int(ends[e["episode"] - 1])
            cut = e["cut_chunk"]["c"]  # 1-based chunk at which (c) cuts
            chunk_ids = list(range(cut - args.patience, cut + 1))  # chunk before the run, then the run
            tiles = [tile(z["data/img"][start], f"ep{e['episode']} f0 {Path(e['zarr']).stem}")]
            for c in chunk_ids:
                if c < 1:
                    tiles.append(np.zeros_like(tiles[0]))
                    continue
                lab = e["labels"][c - 1]
                frame = z["data/img"][start + c * args.n_act - 1]
                tiles.append(tile(frame, f"chunk {c}: {lab}"))
            rows.append(np.concatenate(tiles, axis=1))
        sheet = np.concatenate(rows, axis=0)
        out = video_dir / f"contact_{folder}.png"
        cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(out, sheet.shape)


if __name__ == "__main__":
    main()

"""Validate the tuned morph-detection thresholds (area_ratio in [0.75,1.20],
min_iou=0.35) against a REAL, already-generated imagined-data pool -- a different,
production rollout distribution (DP-policy-driven, not the synthetic random-walk
used to tune the thresholds), as an out-of-sample check per user request.

Usage:
  MUJOCO_GL=egl /home/jacobhb/RLinf/.venv/bin/python \
      rlinf_integration/validate_morph_detection_on_pool.py \
      --zarr datasets/scaling_v3/imag/half1.zarr --n_episodes 20
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from collect_imagined_rotate_t import make_templates, est_angle_with_conf, red_mask  # noqa: E402

REWARD_THETAS = np.radians(np.arange(-180, 180, 1.5))
AREA_LOW, AREA_HIGH, MIN_IOU = 0.75, 1.20, 0.35


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="datasets/scaling_v3/imag/half1.zarr")
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--stride", type=int, default=8, help="check every Nth frame (8 = chunk cadence)")
    ap.add_argument("--out_dir", default="outputs/morph_detection_validation")
    args = ap.parse_args()

    z = zarr.open(args.zarr, mode="r")
    ends = z["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    n_avail = len(ends)
    picks = np.arange(min(args.n_episodes, n_avail))
    print(f"Pool: {args.zarr}  ({n_avail} episodes total, checking {len(picks)})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_ep_flagged = []
    total_frames_checked = 0
    total_flagged = 0
    flagged_events = []  # (ep, frame_idx, area_ratio, iou)

    for ep in picks:
        s, e = starts[ep], ends[ep]
        imgs = z["data/img"][s:e]
        L = len(imgs)
        templates, tc = make_templates(imgs[0], thetas=REWARD_THETAS)
        f0_area = red_mask(imgs[0]).sum()
        flagged_this_ep = 0
        checked_this_ep = 0
        first_flag_frame = None
        for t in range(0, L, args.stride):
            angle, iou, area = est_angle_with_conf(imgs[t], templates, tc, thetas=REWARD_THETAS)
            checked_this_ep += 1
            ratio = area / f0_area
            is_morph = ratio > AREA_HIGH
            if is_morph:
                flagged_this_ep += 1
                if first_flag_frame is None:
                    first_flag_frame = t
                flagged_events.append((int(ep), t, float(ratio), float(iou)))
        total_frames_checked += checked_this_ep
        total_flagged += flagged_this_ep
        per_ep_flagged.append(flagged_this_ep)
        flag_str = f"FLAGGED at frame {first_flag_frame}" if flagged_this_ep else "clean"
        print(f"  ep{ep:03d}: L={L:4d}  checked={checked_this_ep:3d}  "
              f"flagged={flagged_this_ep:3d}  [{flag_str}]")

    print(f"\nTotal: {total_flagged}/{total_frames_checked} frame-checks flagged "
          f"({100*total_flagged/total_frames_checked:.1f}%) across {len(picks)} episodes")
    n_eps_with_any_flag = sum(1 for c in per_ep_flagged if c > 0)
    print(f"Episodes with >=1 flagged frame: {n_eps_with_any_flag}/{len(picks)}")

    if flagged_events:
        np.save(out_dir / "flagged_events.npy", np.array(flagged_events, dtype=object), allow_pickle=True)
        print(f"Flagged event details saved to {out_dir / 'flagged_events.npy'}")
        # save a few flagged frames as images for visual spot-check
        import cv2
        n_saved = 0
        for (ep, t, ratio, iou) in flagged_events[:12]:
            s = starts[ep]
            img = z["data/img"][s + t]
            im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.putText(im, f"ep{ep} t{t} ratio={ratio:.2f} iou={iou:.2f}", (4, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imwrite(str(out_dir / f"flagged_ep{ep:03d}_t{t:03d}.png"), im)
            n_saved += 1
        print(f"Saved {n_saved} flagged-frame images to {out_dir}/ for visual spot-check")
    else:
        print("No frames flagged -- nothing to spot-check.")


if __name__ == "__main__":
    main()

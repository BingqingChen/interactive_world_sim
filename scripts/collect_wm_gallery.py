"""Render the videos behind the world-model half of the scaling study.

Two galleries, both written as small mp4s plus a manifest for the browser page:

paired    The 24 evaluation episodes from `outputs/wm_quality/rollout_frames.npz`,
          each as a side-by-side clip: the world model's imagined rollout on the
          left, the simulator replaying the identical commanded actions on the
          right. This is what `outputs/wm_video_metrics_plot.png` quantifies, so
          the per-episode metrics (PSNR, flow AEPE, subject consistency, final
          T angle) travel with each clip.

invisible Demos from `datasets/rotate_t_invis`, where the T block is rendered with
          alpha=0 while its physics is untouched -- the arms really are rotating a
          block you cannot see. Used to test whether a policy can learn the motion
          without ever observing the object.

imagined  Episodes from the imagined training pool itself -- randA(375) +
          randB(375) + randC(50) as pooled indices 0..799, exactly as
          `build_scaling_zarrs.py` slices them. A dose of +N takes pooled 0:N, so
          each sampled episode is tagged with the doses that contain it. These are
          the demos the +50/+400/+800 policies actually trained on.

Videos use the same convention as the policy rollouts: every 2nd control step at
5 fps, so real-time duration is preserved and the page's speed buttons stay exact.

Usage:
    python scripts/collect_wm_gallery.py                 # both galleries
    python scripts/collect_wm_gallery.py --mode imagined --n_imagined 12
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parent.parent
WM_QUALITY = REPO_ROOT / "outputs" / "wm_quality"
DATASETS = REPO_ROOT / "datasets"
CONTROL_HZ = 10

# Pooled imagined index ranges, in the order build_scaling_zarrs.py concatenates them.
IMAGINED_POOLS = [
    ("randA", "rotate_t_imagined_randA_dp.zarr"),
    ("randB", "rotate_t_imagined_randB_dp.zarr"),
    ("randC", "rotate_t_imagined_randC_dp.zarr"),
]
DOSES = [50, 400, 800]


def write_video(
    frames: npt.NDArray[np.uint8], path: Path, stride: int, fps: float
) -> int:
    """Write RGB frames to mp4, keeping real-time duration. Returns frame count."""
    import imageio.v2 as imageio

    sub = frames[::stride]
    imageio.mimwrite(
        str(path),
        sub,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        output_params=["-crf", "26"],
    )
    return len(sub)


def yaw(quat: npt.NDArray[np.float64]) -> float:
    """Z-rotation of a (w, x, y, z) quaternion, matching collect_rotate_t.t_angle."""
    import transforms3d

    m = transforms3d.quaternions.quat2mat(quat)
    return float(np.arctan2(m[1, 0], m[0, 0]))


def wrap_pi(theta: float) -> float:
    """Wrap an angle in radians into [-pi, pi)."""
    return float((theta + np.pi) % (2 * np.pi) - np.pi)


def divider(h: int, w: int = 2) -> npt.NDArray[np.uint8]:
    """A thin vertical separator so the two panes are visually distinct."""
    return np.full((h, w, 3), 40, dtype=np.uint8)


def build_paired(out_root: Path, stride: int, fps: float) -> List[Dict[str, Any]]:
    """Side-by-side world-model vs simulator clips for the 24 eval episodes."""
    frames_path = WM_QUALITY / "rollout_frames.npz"
    if not frames_path.exists():
        print(f"  {frames_path} missing; skipping paired gallery")
        return []

    frames = np.load(frames_path)
    wm_all, sim_all = frames["wm"], frames["sim"]
    roll = np.load(WM_QUALITY / "rollout_metrics.npz")
    vid = (
        np.load(WM_QUALITY / "video_metrics.npz")
        if (WM_QUALITY / "video_metrics.npz").exists()
        else None
    )

    out_dir = out_root / "paired"
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes: List[Dict[str, Any]] = []

    for i in range(len(wm_all)):
        wm, sim = wm_all[i], sim_all[i]
        bar = divider(wm.shape[1])
        combined = np.concatenate(
            [wm, np.repeat(bar[None], len(wm), axis=0), sim], axis=2
        )
        name = f"pair{i:02d}.mp4"
        n = write_video(combined, out_dir / name, stride, fps)

        entry: Dict[str, Any] = {
            "index": i,
            "video": f"paired/{name}",
            "frames": n,
            "duration_s": round(len(wm) / CONTROL_HZ, 1),
            "psnr_mean": round(float(roll["psnr"][i].mean()), 2),
            "psnr_final": round(float(roll["psnr"][i][-1]), 2),
            "lpips_mean": round(float(roll["lpips"][i].mean()), 4),
            "wm_angle_final": round(float(np.degrees(roll["wm_angle"][i][-1])), 1),
            "gt_angle_final": round(float(np.degrees(roll["gt_angle"][i][-1])), 1),
            "wm_crossed": bool(roll["wm_crossed"][i]),
            "sim_crossed": bool(roll["sim_crossed"][i]),
        }
        if vid is not None:
            entry["subject_consistency"] = round(float(vid["sc_wm_rollout"][i]), 4)
            entry["aepe_mean"] = round(float(vid["aepe_color_rollout"][i].mean()), 4)
        episodes.append(entry)
        print(
            f"  pair {i:02d}: psnr {entry['psnr_mean']:5.2f}  "
            f"wm {entry['wm_angle_final']:+6.1f} "
            f"sim {entry['gt_angle_final']:+6.1f}",
            flush=True,
        )
    return episodes


def doses_containing(pooled_idx: int) -> List[int]:
    """Which +N imagined doses include this pooled episode (a dose takes 0:N)."""
    return [d for d in DOSES if pooled_idx < d]


def build_imagined(
    out_root: Path, n_samples: int, stride: int, fps: float
) -> List[Dict[str, Any]]:
    """Sample episodes from the pooled imagined training data and render them."""
    import zarr

    pools = []
    offset = 0
    for label, rel in IMAGINED_POOLS:
        path = DATASETS / rel
        if not path.exists():
            print(f"  {rel} missing; skipping")
            continue
        group = zarr.open(str(path), "r")
        ends = np.asarray(group["meta/episode_ends"][:])
        pools.append((label, group, ends, offset))
        offset += len(ends)
    if not pools:
        return []
    total = offset

    out_dir = out_root / "imagined"
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = np.unique(np.linspace(0, total - 1, n_samples).astype(int))
    episodes: List[Dict[str, Any]] = []

    for pooled in picks:
        label, group, ends, base = next(p for p in reversed(pools) if pooled >= p[3])
        local = int(pooled - base)
        start = int(ends[local - 1]) if local > 0 else 0
        stop = int(ends[local])
        imgs = np.asarray(group["data/img"][start:stop])

        name = f"imag{int(pooled):03d}.mp4"
        n = write_video(imgs, out_dir / name, stride, fps)
        episodes.append(
            {
                "pooled_index": int(pooled),
                "pool": label,
                "video": f"imagined/{name}",
                "frames": n,
                "steps": stop - start,
                "duration_s": round((stop - start) / CONTROL_HZ, 1),
                "in_doses": doses_containing(int(pooled)),
            }
        )
        print(
            f"  imagined pooled={pooled:3d} ({label}) {stop - start:3d} steps "
            f"-> doses {doses_containing(int(pooled))}",
            flush=True,
        )
    return episodes


def build_invisible(
    out_root: Path, n_samples: int, stride: int, fps: float
) -> List[Dict[str, Any]]:
    """Render demos in which the T is invisible to the camera but not to physics."""
    import h5py

    src = DATASETS / "rotate_t_invis"
    if not src.exists():
        print(f"  {src} missing; skipping")
        return []
    files = sorted(src.glob("episode_*.hdf5"), key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        return []
    picks = [
        files[i]
        for i in np.unique(np.linspace(0, len(files) - 1, n_samples).astype(int))
    ]

    out_dir = out_root / "invisible"
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes: List[Dict[str, Any]] = []
    for path in picks:
        with h5py.File(path, "r") as f:
            imgs = np.asarray(f["obs"]["images"]["top_pov"][()])
            state = np.asarray(f["env_state"][()])
        # The block is unrendered, but its pose is still recorded, so the rotation it
        # actually underwent can be reported alongside a video that never shows it.
        # Same convention as collect_rotate_t.t_angle; clockwise reported positive.
        delta_deg = -np.degrees(wrap_pi(yaw(state[-1][3:]) - yaw(state[0][3:])))
        name = f"{path.stem}.mp4"
        n = write_video(imgs, out_dir / name, stride, fps)
        episodes.append(
            {
                "episode": path.stem,
                "video": f"invisible/{name}",
                "frames": n,
                "steps": int(len(imgs)),
                "duration_s": round(len(imgs) / CONTROL_HZ, 1),
                "rotation_deg": round(float(delta_deg), 1),
            }
        )
        print(
            f"  {path.stem}: {len(imgs):3d} steps, "
            f"hidden T turned {episodes[-1]['rotation_deg']:.0f} deg",
            flush=True,
        )
    return episodes


def main() -> None:
    """Render the requested galleries and write outputs/wm_gallery/manifest.json."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode", default="all", choices=["all", "paired", "imagined", "invisible"]
    )
    ap.add_argument("--n_invisible", type=int, default=5)
    ap.add_argument("--out_dir", default="outputs/wm_gallery")
    ap.add_argument("--n_imagined", type=int, default=12)
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth control step")
    ap.add_argument("--fps", type=float, default=5.0)
    args = ap.parse_args()

    out_root = REPO_ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)

    manifest: Dict[str, Any] = {
        "protocol": {
            "fps": args.fps,
            "control_hz": CONTROL_HZ,
            "stride": args.stride,
            "note": (
                "Videos keep real-time duration (every Nth control step at a "
                "correspondingly lower fps), so playback speed is an exact multiple."
            ),
        },
        "summary": {},
        "paired": [],
        "imagined": [],
        "invisible": [],
    }
    summary_path = WM_QUALITY / "video_metrics.json"
    if summary_path.exists():
        manifest["summary"] = json.loads(summary_path.read_text())

    # Seed from any existing manifest so that regenerating a single gallery leaves the
    # other one intact -- otherwise `--mode imagined` silently drops the paired clips
    # (which stay on disk) from the page.
    existing_path = out_root / "manifest.json"
    if existing_path.exists():
        prev = json.loads(existing_path.read_text())
        for key in ("paired", "imagined", "invisible"):
            manifest[key] = prev.get(key, [])

    if args.mode in ("all", "paired"):
        print("== paired world-model vs simulator ==")
        manifest["paired"] = build_paired(out_root, args.stride, args.fps)
    if args.mode in ("all", "imagined"):
        print("== imagined training demos ==")
        manifest["imagined"] = build_imagined(
            out_root, args.n_imagined, args.stride, args.fps
        )
    if args.mode in ("all", "invisible"):
        print("== invisible-T demos ==")
        manifest["invisible"] = build_invisible(
            out_root, args.n_invisible, args.stride, args.fps
        )

    path = out_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(
        f"\nwrote {path}  "
        f"({len(manifest['paired'])} paired, {len(manifest['imagined'])} imagined)"
    )


if __name__ == "__main__":
    main()

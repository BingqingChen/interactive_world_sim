"""Re-encode rollout videos down to the frames' native resolution.

Earlier runs of `collect_scaling_rollouts.py` upscaled 128x128 frames to 256x256 before
encoding, which roughly tripled the file size for no extra information -- and made the
scaling page slow to load, since selecting a condition fetches ten videos at once over
an SSH port-forward. This rewrites any oversized video in place at native size.

Idempotent: files already at the target width are skipped, so it is safe to re-run
(including while a collection sweep is still writing new files).

Usage:
    python scripts/shrink_rollout_videos.py [--dir DIR] [--width 128]
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import av

REPO_ROOT = Path(__file__).resolve().parent.parent


def video_info(path: Path) -> tuple:
    """(width, fps) of a video file, or (0, 0.0) if it cannot be read."""
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            return int(stream.codec_context.width), float(stream.average_rate)
    except Exception:
        return 0, 0.0


def shrink(path: Path, width: int, fps: float, crf: int, src_fps: float) -> tuple:
    """Re-encode one file at `width` and `fps`, replacing it only if ffmpeg succeeds.

    Frames are dropped by an integer stride and the frame rate is lowered to match, so
    the clip keeps its true real-time duration -- the speed buttons on the page stay
    exact multiples of real time.
    """
    tmp = path.with_suffix(".tmp.mp4")
    before = path.stat().st_size
    stride = max(1, round(src_fps / fps)) if fps > 0 else 1
    vf = f"scale={width}:{width}:flags=area"
    if stride > 1:
        vf = f"select='not(mod(n\\,{stride}))',setpts=N/({fps}*TB)," + vf
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        vf,
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return before, before, False
    after = tmp.stat().st_size
    tmp.replace(path)
    return before, after, True


def main() -> None:
    """Shrink every oversized rollout video under the given directory."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="outputs/scaling_rollouts")
    ap.add_argument("--width", type=int, default=128)
    # 5 fps = every 2nd control step. Real-time duration is preserved, and at the page's
    # default 4x that is still 20 effective fps. Halves the bytes again on top of the
    # resolution fix, which matters because selecting a condition fetches ten files.
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--crf", type=int, default=26)
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")

    root = REPO_ROOT / args.dir
    # rglob, so --dir can be the rollout root or a single condition directory (useful
    # for shrinking finished conditions while a sweep is still writing later ones).
    videos = sorted(p for p in root.rglob("*.mp4") if not p.name.endswith(".tmp.mp4"))
    if not videos:
        raise SystemExit(f"no videos under {root}")

    saved = kept = failed = 0
    total_before = total_after = 0
    for path in videos:
        w, fps = video_info(path)
        if w == 0:
            failed += 1
            print(f"  unreadable, skipped: {path.name}")
            continue
        if w <= args.width and fps <= args.fps + 1e-6:
            kept += 1
            total_before += path.stat().st_size
            total_after += path.stat().st_size
            continue
        before, after, ok = shrink(path, args.width, args.fps, args.crf, fps)
        total_before += before
        total_after += after
        if ok:
            saved += 1
        else:
            failed += 1
            print(f"  ffmpeg failed, left as-is: {path.name}")

    mb = 1024 * 1024
    print(
        f"\n{len(videos)} videos: {saved} shrunk, {kept} native, {failed} failed\n"
        f"{total_before / mb:.1f} MB -> {total_after / mb:.1f} MB "
        f"({100 * (1 - total_after / max(total_before, 1)):.0f}% smaller)"
    )


if __name__ == "__main__":
    main()

"""Build the three Diffusion Policy zarr replay buffers for the invisible-cube Lift ablation.

    centre  = 50 successful demos, cube in the stock +/-3 cm square
    vis     = those same 50 + 400 successful demos from the +/-10 cm square
    invis   = those same 50 + the SAME 400, re-rendered with the cube hidden

The comparison only means anything if `vis` and `invis` differ in pixels and nothing else, so the
selected 400 episode indices are written to a JSON manifest on the first build and re-read (never
re-derived) afterwards. `--which invis` refuses to run if the manifest is missing: silently
selecting its own 400 is the one failure mode that would invalidate the experiment while looking
completely healthy.

Zarr keys, matching what diffusion_policy's image datasets consume:
    img     (N, 128, 128, 3) uint8   -- agentview, resized from the stored 256 px with INTER_AREA
    wrist   (N, 128, 128, 3) uint8   -- robot0_eye_in_hand, same resize
    state   (N, 8)           float32 -- eef_pos(3) + eef axis-angle(3) + gripper_qpos(2)
    action  (N, 7)           float32 -- the recorded OSC_POSE delta actually executed
plus meta/episode_ends.

Usage (IWS venv -- needs zarr + cv2, not robosuite):
    .venv/bin/python scripts/lift_invis/build_lift_zarrs.py --which centre
    .venv/bin/python scripts/lift_invis/build_lift_zarrs.py --which vis
    .venv/bin/python scripts/lift_invis/build_lift_zarrs.py --which invis
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

DP_ROOT = "/home/jacobhb/projects/worth_doing/diffusion_policy"
sys.path.insert(0, DP_ROOT)
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402

OPENPI = Path("/home/jacobhb/projects/openpi/data/robosuite")
CENTRE_DIR = OPENPI / "rollouts_lift_pi" / "lift"
LARGE_DIRS = [
    OPENPI / "rollouts_lift_large" / "lift_large_pilot",
    OPENPI / "rollouts_lift_large" / "lift_large",
]
INVIS_DIR = OPENPI / "rollouts_lift_large" / "lift_large_invis"
CENTRE_INVIS_DIR = OPENPI / "rollouts_lift_centre_invis"

OUT_DIR = Path("/home/jacobhb/projects/worth_doing/interactive_world_sim/datasets")
MANIFEST = OUT_DIR / "lift_invis_manifest.json"

RESOLUTION = 128
N_CENTRE = 50
N_LARGE = 400


def ep_index(p: Path) -> int:
    return int(p.stem.split("_")[-1])


def is_success(p: Path) -> bool:
    """False for episodes still being written (savez_compressed is not atomic)."""
    try:
        return bool(np.load(p, allow_pickle=True)["success"])
    except (zipfile.BadZipFile, EOFError, KeyError, ValueError):
        return False


def successes_in(dirs: list[Path]) -> list[Path]:
    """All successful episodes across `dirs`, ordered by episode index (which is globally unique)."""
    files: list[Path] = []
    for d in dirs:
        files += [Path(p) for p in glob.glob(str(d / "episode_*.npz"))]
    return sorted((f for f in files if is_success(f)), key=ep_index)


def resize(imgs: np.ndarray) -> np.ndarray:
    """(T, 256, 256, 3) -> (T, 128, 128, 3), INTER_AREA (same as the IWS HDF5 converter)."""
    if imgs.shape[1] == RESOLUTION and imgs.shape[2] == RESOLUTION:
        return imgs.astype(np.uint8)
    return np.stack(
        [cv2.resize(f, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA) for f in imgs],
        axis=0,
    ).astype(np.uint8)


def add(rb: ReplayBuffer, path: Path) -> int:
    d = np.load(path, allow_pickle=True)
    img, wrist = resize(d["agentview"]), resize(d["wrist"])
    state, action = d["state"].astype(np.float32), d["actions"].astype(np.float32)

    # The collector appends images, state and action in the same iteration, so all four must
    # already agree; truncate to the shortest rather than trusting that.
    n = min(len(img), len(wrist), len(state), len(action))
    assert n > 0, f"{path}: empty episode"
    assert state.shape[1] == 8 and action.shape[1] == 7, f"{path}: dims {state.shape} {action.shape}"

    rb.add_episode(
        {"img": img[:n], "wrist": wrist[:n], "state": state[:n], "action": action[:n]}
    )
    return n


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(m: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2))


def pick_centre() -> list[int]:
    """The first N_CENTRE successes of the existing +/-3 cm set."""
    files = successes_in([CENTRE_DIR])
    if len(files) < N_CENTRE:
        raise RuntimeError(f"only {len(files)} centre successes, need {N_CENTRE}")
    return [ep_index(f) for f in files[:N_CENTRE]]


def pick_large() -> list[int]:
    """The first N_LARGE successes of the +/-10 cm set (pilot + main pooled, index order)."""
    files = successes_in(LARGE_DIRS)
    if len(files) < N_LARGE:
        raise RuntimeError(
            f"only {len(files)} large-region successes, need {N_LARGE} -- collection unfinished"
        )
    return [ep_index(f) for f in files[:N_LARGE]]


def find(idx: int, dirs: list[Path]) -> Path:
    for d in dirs:
        p = d / f"episode_{idx:04d}.npz"
        if p.exists():
            return p
    raise FileNotFoundError(f"episode_{idx:04d}.npz not in {[str(d) for d in dirs]}")


# Each condition is a list of (source, start, stop) parts naming a HALF-OPEN SLICE of a manifest
# list. Slices (not just prefixes) are required because a condition can draw the same manifest list
# both visible and hidden -- e.g. `spread` shows the first 20 in-region episodes and hides the
# remaining 30. Those slices must be disjoint: an episode used both visible and hidden would put
# two contradictory renderings of identical physics into one dataset. `assert_disjoint` enforces it.
#
# Slicing from shared manifest lists also keeps conditions nested, so a difference between two
# conditions is a difference in what was ADDED, not in which sample was drawn.
#
#   centre       = in-region demos, cube visible             (rollouts_lift_pi/lift, +/-3 cm)
#   centre_invis = the SAME in-region demos, cube hidden     (rollouts_lift_centre_invis)
#   large_vis    = out-of-region demos, cube visible         (rollouts_lift_large/*, +/-10 cm)
#   large_invis  = the SAME out-of-region demos, cube hidden (lift_large_invis)
COMPOSITIONS: dict[str, list[tuple[str, int, int]]] = {
    "centre": [("centre", 0, 50)],
    "vis": [("centre", 0, 50), ("large_vis", 0, 400)],
    "invis": [("centre", 0, 50), ("large_invis", 0, 400)],
    # 20 in-region + 20 out-of-region VISIBLE: does a handful of visible out-of-region demos buy
    # the generalization that 400 did?
    "s20v20": [("centre", 0, 20), ("large_vis", 0, 20)],
    # 30 in-region + 380 out-of-region HIDDEN. Superseded by `spread` and not trained to
    # completion; kept so the composition is on record.
    "s30h380": [("centre", 0, 30), ("large_invis", 0, 380)],
    # THE VISIBLE DEMOS SPAN BOTH REGIONS: 20 visible in-region + 30 visible out-of-region, with
    # every remaining episode hidden. Same total (450) and the same 50-visible/400-hidden split as
    # `invis`; the ONLY difference is where the visible demos sit. So it asks whether hidden data
    # helps once the policy can already localize the cube everywhere -- i.e. whether the failure of
    # `invis` out of region was purely a localization failure.
    "spread": [
        ("centre", 0, 20),          # visible, in-region
        ("large_vis", 0, 30),       # visible, out-of-region
        ("centre_invis", 20, 50),   # hidden, in-region  (the 30 not shown visible)
        ("large_invis", 30, 400),   # hidden, out-of-region (the 370 not shown visible)
    ],
}

SOURCE_DIRS = {
    "centre": [CENTRE_DIR],
    "centre_invis": [CENTRE_INVIS_DIR],
    "large_vis": LARGE_DIRS,
    "large_invis": [INVIS_DIR],
}
MANIFEST_KEY = {
    "centre": "centre",
    "centre_invis": "centre",
    "large_vis": "large",
    "large_invis": "large",
}


def assert_disjoint(which: str, parts: list[tuple[str, int, int]], man: dict) -> None:
    """No episode may appear twice in one condition, whether or not the renderings differ.

    Two parts collide when they slice overlapping ranges of the SAME manifest list -- which is
    exactly how a visible and a hidden copy of one episode would both end up in the dataset.
    """
    seen: dict[tuple[str, int], str] = {}
    for src, start, stop in parts:
        key = MANIFEST_KEY[src]
        for idx in man[key][start:stop]:
            prev = seen.get((key, idx))
            if prev is not None:
                raise RuntimeError(
                    f"{which}: episode {idx} of '{key}' appears in both '{prev}' and '{src}' -- "
                    f"the same physics would enter the dataset twice with different pixels"
                )
            seen[(key, idx)] = src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=sorted(COMPOSITIONS), required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    man = load_manifest()

    # The 50 centre indices are fixed on first build and shared by every condition.
    if "centre" not in man:
        man["centre"] = pick_centre()
        save_manifest(man)

    parts = COMPOSITIONS[args.which]
    needs_large = any(MANIFEST_KEY[src] == "large" for src, _, _ in parts)
    if needs_large and "large" not in man:
        # Only a VISIBLE build may create the large list; an invisible-only build must not, or the
        # two could silently end up on different episode sets.
        if any(src == "large_vis" for src, _, _ in parts):
            man["large"] = pick_large()
            save_manifest(man)
        else:
            raise RuntimeError(
                "manifest has no 'large' list -- build a visible condition (vis / s20v20) first "
                "so hidden and visible conditions provably use the SAME episodes"
            )

    assert_disjoint(args.which, parts, man)

    paths: list[Path] = []
    for src, start, stop in parts:
        pool = man[MANIFEST_KEY[src]]
        if len(pool) < stop:
            raise RuntimeError(
                f"{args.which}: need {MANIFEST_KEY[src]}[{start}:{stop}], manifest has {len(pool)}"
            )
        paths += [find(i, SOURCE_DIRS[src]) for i in pool[start:stop]]
    print("composition: " + " + ".join(f"{stop - start} {src}" for src, start, stop in parts))

    out = args.out or OUT_DIR / f"lift_{args.which}_dp.zarr"
    print(f"which={args.which}  episodes={len(paths)}  -> {out}")

    rb = ReplayBuffer.create_empty_numpy()
    total = 0
    for k, p in enumerate(paths):
        total += add(rb, p)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(paths)} episodes, {total} steps", flush=True)

    rb.save_to_path(str(out), if_exists="replace")
    lens = np.diff(np.r_[0, rb.episode_ends[:]])
    print(f"wrote {rb.n_episodes} episodes / {total} steps")
    print(f"  img{rb['img'].shape} wrist{rb['wrist'].shape} "
          f"state{rb['state'].shape} action{rb['action'].shape}")
    print(f"  episode length: min {lens.min()} max {lens.max()} mean {lens.mean():.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

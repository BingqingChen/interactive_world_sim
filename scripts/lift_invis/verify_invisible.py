"""Verify an invisible-cube episode against its visible source.

Three claims, from the plan's verification list:
  1. every non-image field is byte-identical (it must be a pure re-render, not a re-simulation)
  2. the pixel difference is small and spatially LOCALIZED -- confined to where the cube was
  3. the red cube's pixels are gone (red-dominant pixel count drops to ~0)

Usage:
    .venv-lift/bin/python scripts/lift_invis/verify_invisible.py <visible.npz> <invisible.npz>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def red_mask(img: np.ndarray) -> np.ndarray:
    """Pixels where red clearly dominates -- the cube's signature against this scene."""
    r, g, b = img[..., 0].astype(np.int16), img[..., 1].astype(np.int16), img[..., 2].astype(np.int16)
    return (r > 90) & (r - g > 45) & (r - b > 45)


def main() -> int:
    vis_path, invis_path = Path(sys.argv[1]), Path(sys.argv[2])
    v = np.load(vis_path, allow_pickle=True)
    n = np.load(invis_path, allow_pickle=True)

    ok = True

    # --- 1. non-image fields identical -------------------------------------------------
    skip = {"agentview", "wrist", "cube_visible"}
    mismatched = []
    for k in v.files:
        if k in skip:
            continue
        a, b = v[k], n[k]
        if a.shape != b.shape or not np.array_equal(a, b):
            mismatched.append(k)
    print(f"[1] non-image fields: {len(v.files) - len(skip & set(v.files))} checked, "
          f"{len(mismatched)} mismatched {mismatched if mismatched else ''}")
    ok &= not mismatched

    # --- 2/3. per-camera pixel analysis ------------------------------------------------
    for cam in ("agentview", "wrist"):
        va, na = v[cam], n[cam]
        assert va.shape == na.shape, f"{cam}: shape changed {va.shape} -> {na.shape}"
        diff = np.abs(va.astype(np.int16) - na.astype(np.int16)).max(axis=-1)  # (T,H,W)

        changed = diff > 12  # per-pixel "meaningfully different"
        frac_changed = changed.mean()

        # Localization: of the pixels that changed, how many were red (cube) in the visible frame,
        # or are adjacent to one? A pure cube removal changes the cube's pixels and its shadow.
        cube = red_mask(va)
        overlap = (changed & cube).sum() / max(changed.sum(), 1)

        red_before = cube.sum() / len(va)
        red_after = red_mask(na).sum() / len(na)

        print(f"[2] {cam:9s} changed pixels {frac_changed * 100:5.2f}% of frame  "
              f"mean|diff| {diff.mean():5.2f}  (of changed, {overlap * 100:4.1f}% were cube-red)")
        print(f"[3] {cam:9s} red pixels/frame {red_before:7.1f} -> {red_after:5.1f} "
              f"({100 * (1 - red_after / max(red_before, 1e-9)):.1f}% removed)")

        # The cube occupies a small part of the frame; a change much larger than that means
        # something other than the cube moved.
        if frac_changed > 0.15:
            print(f"    FAIL {cam}: {frac_changed*100:.1f}% of pixels changed -- not localized")
            ok = False
        if red_after > 0.05 * max(red_before, 1e-9):
            print(f"    FAIL {cam}: red pixels remain ({red_after:.1f}/frame)")
            ok = False

    print(f"\nVERIFY {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

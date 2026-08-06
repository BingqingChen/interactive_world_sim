"""Frame metrics for the lockstep sim-vs-world-model browser demo.

Compares a world-model-predicted 128x128 RGB frame against the ground-truth MuJoCo
render produced by the identical commanded action, and estimates the T block's
rotation from either image.

Everything here is pure numpy/OpenCV on uint8 (H, W, 3) RGB arrays -- no torch, no
MuJoCo, no GPU -- so it can be exercised standalone.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

# Full-circle template grid. The estimator this is lifted from
# (scripts/collect_imagined_rotate_t.py:66) used arange(-130, 41, 1.5) because its task
# only ever rotated ~90 degrees clockwise; free teleop turns either way, so a truncated
# grid would silently saturate at its endpoints.
THETAS: npt.NDArray[np.float64] = np.radians(np.arange(-180.0, 180.0, 1.5))

# Below this many red pixels the T is considered lost (the world model has drifted far
# enough that the block is no longer renderable).
MIN_T_PIXELS = 30


def red_mask(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
    """Segment the red T block from a (H, W, 3) uint8 RGB frame."""
    # lifted from scripts/collect_imagined_rotate_t.py:69-74
    r = img[..., 0].astype(np.int16)
    g = img[..., 1].astype(np.int16)
    b = img[..., 2].astype(np.int16)
    return (r > 140) & (g < 110) & (b < 110) & (r - np.maximum(g, b) > 40)


def make_templates(
    frame0: npt.NDArray[np.uint8],
) -> Tuple[npt.NDArray[np.bool_], Tuple[float, float]]:
    """Build the rotated-mask template bank from an episode's first frame.

    The T's mask in ``frame0`` is rotated over ``THETAS`` about its own centroid, so
    every later estimate is *relative to frame 0* -- frame 0 need not be upright.

    Returns:
        (templates (n_theta, H, W) bool, centroid (cy, cx)).
    """
    # lifted from scripts/collect_imagined_rotate_t.py:77-88
    m0 = red_mask(frame0).astype(np.uint8)
    ys, xs = np.nonzero(m0)
    if len(ys) < MIN_T_PIXELS:
        raise ValueError(
            f"frame 0 has only {len(ys)} red pixels; cannot build T-angle templates"
        )
    cy, cx = float(ys.mean()), float(xs.mean())
    tpl = np.stack(
        [
            cv2.warpAffine(
                m0,
                cv2.getRotationMatrix2D((cx, cy), np.degrees(th), 1.0),
                (m0.shape[1], m0.shape[0]),
            )
            > 0
            for th in THETAS
        ]
    )
    return tpl, (cy, cx)


def est_angle(
    img: npt.NDArray[np.uint8],
    templates: npt.NDArray[np.bool_],
    tc: Tuple[float, float],
) -> Tuple[Optional[float], int]:
    """Estimate the T's rotation (rad, relative to frame 0) by best-IoU template match.

    Returns:
        (angle or None if the T is not visible, red-mask pixel count). The pixel count
        is returned even when the angle is None so the caller can watch the mask decay
        before it crosses the detection floor.
    """
    # lifted from scripts/collect_imagined_rotate_t.py:91-105, extended to also
    # report the mask size
    m = red_mask(img)
    n_pixels = int(m.sum())
    if n_pixels < MIN_T_PIXELS:
        return None, n_pixels
    ys, xs = np.nonzero(m)
    m_al = (
        cv2.warpAffine(
            m.astype(np.uint8),
            np.float32([[1, 0, tc[1] - xs.mean()], [0, 1, tc[0] - ys.mean()]]),
            (m.shape[1], m.shape[0]),
        )
        > 0
    )
    inter = (templates & m_al).sum(axis=(1, 2))
    union = (templates | m_al).sum(axis=(1, 2))
    return float(THETAS[np.argmax(inter / np.maximum(union, 1))]), n_pixels


def wrap_angle(theta: float) -> float:
    """Wrap an angle in radians into [-pi, pi)."""
    return float((theta + np.pi) % (2 * np.pi) - np.pi)


def mse(a: npt.NDArray[np.uint8], b: npt.NDArray[np.uint8]) -> float:
    """Mean squared error between two uint8 frames, in squared 0-255 units."""
    return float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())


def psnr(a: npt.NDArray[np.uint8], b: npt.NDArray[np.uint8]) -> float:
    """Peak signal-to-noise ratio (dB) between two uint8 frames."""
    # lifted from scripts/eval_wm_quality.py:72-75, specialized to a single frame pair
    return float(10 * np.log10(255.0**2 / max(mse(a, b), 1e-12)))


def diff_image(
    a: npt.NDArray[np.uint8], b: npt.NDArray[np.uint8], gain: float = 3.0
) -> npt.NDArray[np.uint8]:
    """False-colour absolute difference between two frames, as (H, W, 3) uint8 RGB."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=2)
    scaled = np.clip(d * gain, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO)
    return np.ascontiguousarray(bgr[:, :, ::-1])

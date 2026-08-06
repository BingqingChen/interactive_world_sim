"""Re-render robosuite Lift episodes from their stored `sim_state`, optionally with the cube hidden.

This is one tool with two jobs:

1. **Fidelity gate** (`--mode check`). The rollouts were collected in openpi's py3.8 venv
   (robosuite 1.4.1 / mujoco 3.2.3 / numpy 1.22). Evaluation and this re-render run in the py3.11
   `.venv-lift`. If the two renderers disagree, every downstream image is subtly off-distribution.
   `check` re-renders with the cube VISIBLE and compares against the stored frames; the plan's gate
   is median PSNR > 35 dB.

2. **Invisible-set generator** (`--mode invisible`). Hides the cube's *visual* geom and re-renders.
   Because every step's full `sim_state` is stored, this is a pure re-render -- no physics is
   re-simulated and no policy is queried -- so `state`, `actions`, `rewards`, `success` and every
   other field are copied through byte-identically and only the pixels change.

Why restoring state is enough: `main_lift.py` records `sim_state[i] = env.sim.get_state().flatten()`
PRE-step, in the same iteration that appends `agentview[i]`, so frame i and state i are the same
instant. `set_state_from_flattened` + `forward()` reproduces that instant exactly (mujoco's render
reads only qpos/qvel-derived body poses, which `forward()` recomputes).

Two orientation facts that are easy to get wrong:
  - robosuite returns camera images OpenGL-flipped; the collector stored `img[::-1]` (upright).
    This script applies the same single flip, so its output is directly comparable to `agentview`.
  - the cube's *visual* geom is `cube_g0_vis`; zeroing its alpha leaves the collision geom
    (`cube_g0`) untouched, so physics and `_check_success()` are unaffected.

Usage (from the IWS repo root):
    .venv-lift/bin/python scripts/lift_invis/replay_lift.py --mode check \
        --episodes data/../rollouts_lift_pi/lift/episode_0000.npz --n-episodes 3
    .venv-lift/bin/python scripts/lift_invis/replay_lift.py --mode invisible \
        -i <dir of visible npz> -o <dir for invisible npz>
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

# robosuite prints a macro warning on import; keep stdout clean for the metrics we print.
os.environ.setdefault("MUJOCO_GL", "egl")

import robosuite as suite  # noqa: E402

CUBE_VIS_GEOM = "cube_g0_vis"
ENV_RESOLUTION = 256  # main_lift.py Args.env_resolution -- the stored frames' native size
CAMERAS = ["agentview", "robot0_eye_in_hand"]


def build_env(max_steps: int = 200, num_steps_wait: int = 10):
    """Rebuild the collection env exactly (main_lift.py:_get_robosuite_env)."""
    return suite.make(
        "Lift",
        robots="Panda",
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        gripper_types="default",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=CAMERAS,
        camera_heights=ENV_RESOLUTION,
        camera_widths=ENV_RESOLUTION,
        control_freq=20,
        horizon=max_steps + num_steps_wait + 1,
        ignore_done=True,
        hard_reset=False,
    )


def set_cube_visible(env, visible: bool) -> None:
    """Show/hide the cube's visual geom. Collision geom and physics are untouched."""
    gid = env.sim.model.geom_name2id(CUBE_VIS_GEOM)
    env.sim.model.geom_rgba[gid][3] = 1.0 if visible else 0.0


def render_state(env, flat_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Restore one recorded instant and render both cameras, upright (as the collector stored them)."""
    # NOTE: use set_state_from_flattened, not `get_state()` + assigning `.flat`. MjSimState exposes
    # flatten()/from_flattened() and has no `flat` setter, so assigning it silently creates an unused
    # attribute and set_state writes the UNMODIFIED state back -- every frame renders the reset pose
    # (measured: ~16 dB against the stored frames, which reads exactly like a renderer mismatch).
    env.sim.set_state_from_flattened(np.asarray(flat_state, dtype=np.float64))
    env.sim.forward()

    agentview = env.sim.render(
        camera_name="agentview", width=ENV_RESOLUTION, height=ENV_RESOLUTION
    )[::-1]
    wrist = env.sim.render(
        camera_name="robot0_eye_in_hand", width=ENV_RESOLUTION, height=ENV_RESOLUTION
    )[::-1]
    return np.ascontiguousarray(agentview), np.ascontiguousarray(wrist)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB between two uint8 images. inf when bit-identical."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def mode_check(env, files: list[Path], max_frames: int) -> int:
    """Re-render VISIBLE and compare to the stored frames. Returns a shell exit code."""
    set_cube_visible(env, True)
    all_agent, all_wrist = [], []

    for f in files:
        d = np.load(f, allow_pickle=True)
        if "agentview" not in d:
            print(f"  {f.name}: no stored images, skipping")
            continue
        sim_state, stored_a, stored_w = d["sim_state"], d["agentview"], d["wrist"]
        n = min(len(sim_state), len(stored_a), max_frames)
        pa, pw = [], []
        for i in range(n):
            a, w = render_state(env, sim_state[i])
            pa.append(psnr(a, stored_a[i]))
            pw.append(psnr(w, stored_w[i]))
        all_agent += pa
        all_wrist += pw
        print(f"  {f.name}: {n} frames  agentview median {np.median(pa):6.2f} dB  "
              f"wrist median {np.median(pw):6.2f} dB")

    if not all_agent:
        print("FAIL: no frames compared")
        return 2

    med_a, med_w = float(np.median(all_agent)), float(np.median(all_wrist))
    min_a, min_w = float(np.min(all_agent)), float(np.min(all_wrist))
    print(f"\nGATE  agentview: median {med_a:.2f} dB (min {min_a:.2f})")
    print(f"GATE  wrist    : median {med_w:.2f} dB (min {min_w:.2f})")
    passed = med_a > 35.0 and med_w > 35.0
    print(f"GATE  {'PASS' if passed else 'FAIL'} (threshold: median > 35 dB on both cameras)")
    return 0 if passed else 1


def mode_invisible(env, files: list[Path], out_dir: Path) -> int:
    """Re-render with the cube hidden; copy every non-image field through unchanged."""
    out_dir.mkdir(parents=True, exist_ok=True)
    set_cube_visible(env, False)

    n_done = n_skipped = 0
    for f in files:
        out_path = out_dir / f.name
        if out_path.exists():
            n_done += 1
            continue
        # Skip episodes still being written -- this is run while collection is in flight.
        try:
            d = np.load(f, allow_pickle=True)
            sim_state = d["sim_state"]
            n = min(len(sim_state), len(d["agentview"]))
        except (zipfile.BadZipFile, EOFError, KeyError, ValueError):
            n_skipped += 1
            continue

        agent = np.empty((n, ENV_RESOLUTION, ENV_RESOLUTION, 3), dtype=np.uint8)
        wrist = np.empty_like(agent)
        for i in range(n):
            agent[i], wrist[i] = render_state(env, sim_state[i])

        # Everything except the two image arrays is carried over verbatim.
        payload = {k: d[k] for k in d.files if k not in ("agentview", "wrist")}
        payload["agentview"] = agent
        payload["wrist"] = wrist
        payload["cube_visible"] = False
        # Write to a temp name and rename, so an interrupted render cannot leave a truncated
        # episode that a later pass would skip as "already done".
        # The temp name must itself end in .npz: savez_compressed silently APPENDS .npz to any
        # path that does not, so a ".partial" suffix would write ".partial.npz" and the rename
        # would then fail on a file that was never created. The pid keeps two concurrent passes
        # over the same output directory from writing the same temp file and renaming a
        # half-interleaved result into place.
        tmp = out_path.with_name(f"{out_path.stem}.partial.{os.getpid()}.npz")
        np.savez_compressed(tmp, **payload)
        tmp.rename(out_path)

        n_done += 1
        print(f"  [{n_done}/{len(files)}] {f.name}: {n} frames -> {out_path.name}", flush=True)

    print(f"\nwrote/kept {n_done} episodes to {out_dir}"
          + (f", skipped {n_skipped} still being written" if n_skipped else ""))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["check", "invisible"], required=True)
    p.add_argument("-i", "--input-dir", type=Path, help="directory of episode_*.npz")
    p.add_argument("-o", "--output-dir", type=Path, help="output dir (invisible mode)")
    p.add_argument("--episodes", nargs="*", type=Path, help="explicit npz files (overrides -i)")
    p.add_argument("--n-episodes", type=int, default=3, help="how many episodes to use (check mode)")
    p.add_argument("--max-frames", type=int, default=40, help="frames per episode (check mode)")
    p.add_argument("--successes-only", action="store_true", help="skip episodes with success=False")
    args = p.parse_args()

    if args.episodes:
        files = sorted(args.episodes)
    elif args.input_dir:
        files = sorted(Path(x) for x in glob.glob(str(args.input_dir / "episode_*.npz")))
    else:
        p.error("need --episodes or -i/--input-dir")

    if args.successes_only:
        def ok(f: Path) -> bool:
            # An in-flight episode is not readable yet; treat it as not-a-success and let a later
            # pass pick it up, rather than crashing the whole render.
            try:
                return bool(np.load(f, allow_pickle=True)["success"])
            except (zipfile.BadZipFile, EOFError, KeyError, ValueError):
                return False

        files = [f for f in files if ok(f)]
    if args.mode == "check":
        files = files[: args.n_episodes]
    if not files:
        print("no episodes matched")
        return 2

    print(f"mode={args.mode}  episodes={len(files)}")
    env = build_env()
    env.reset()  # builds the sim + render context; the state is overwritten per frame anyway

    if args.mode == "check":
        return mode_check(env, files, args.max_frames)
    if not args.output_dir:
        p.error("invisible mode needs -o/--output-dir")
    return mode_invisible(env, files, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())

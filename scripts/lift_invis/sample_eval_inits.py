"""Sample the fixed evaluation initial states for the invisible-cube Lift ablation, ONCE.

Writes 50 states with the cube in the +/-3 cm square and 50 in the +/-10 cm square. All three
policies are evaluated on exactly these states, so this file is the experiment's control: it must
be generated once and never regenerated. The script refuses to overwrite an existing output
unless --force is passed.

Two details that make the eval comparable to training:

  - **Post-settle states.** The collector spends `num_steps_wait` (10) steps holding still while
    the dropped cube settles, and only then starts recording. Storing the state *after* those
    same 10 dummy steps means evaluation begins exactly where the training demos begin. Storing
    the raw post-reset state instead would start every episode mid-drop.
  - **Disjoint from training.** Episode indices >= 100000 are used here. Training used seed 7
    (indices 0-999, the +/-3 cm set) and seed 11 (indices 0-1899, the +/-10 cm set) with
    `seed*100003 + idx`; this script uses seed 7 with idx >= 100000, so no eval state can
    coincide with a training state.

Usage:
    .venv-lift/bin/python scripts/lift_invis/sample_eval_inits.py \
        -o datasets/lift_invis_eval_inits.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from replay_lift import build_env  # noqa: E402  -- the same env spec the collector used

DUMMY_ACTION = [0.0] * 6 + [-1.0]  # hold still, gripper open (main_lift.py)
NUM_STEPS_WAIT = 10
EVAL_SEED = 7
EVAL_IDX_BASE = 100_000  # >= this is guaranteed disjoint from every training episode index
N_PER_REGION = 50
REGIONS = {"small": 0.03, "large": 0.10}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", type=Path,
                    default=Path("datasets/lift_invis_eval_inits.npz"))
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing file (defeats the point; be sure)")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        print(f"REFUSING: {args.out} already exists. These states are the experiment's control "
              f"and must not be resampled. Pass --force only if you mean it.")
        return 1

    env = build_env()
    out: dict[str, np.ndarray] = {}

    idx = EVAL_IDX_BASE
    for region, half_width in REGIONS.items():
        env.placement_initializer.x_range = [-half_width, half_width]
        env.placement_initializer.y_range = [-half_width, half_width]

        states, cube_xy, seeds = [], [], []
        for _ in range(N_PER_REGION):
            ep_seed = EVAL_SEED * 100003 + idx
            # The sampler draws cube x/y/yaw from the GLOBAL numpy RNG on every reset, so seeding
            # here is what makes the initial state a pure function of the index.
            np.random.seed(ep_seed)
            obs = env.reset()
            for _ in range(NUM_STEPS_WAIT):
                obs, _, _, _ = env.step(DUMMY_ACTION)

            states.append(np.asarray(env.sim.get_state().flatten(), dtype=np.float64))
            cube_xy.append(np.asarray(obs["cube_pos"][:2], dtype=np.float32))
            seeds.append(ep_seed)
            idx += 1

        s = np.stack(states)
        c = np.stack(cube_xy)
        out[f"{region}_sim_state"] = s
        out[f"{region}_cube_xy"] = c
        out[f"{region}_seed"] = np.asarray(seeds, dtype=np.int64)
        print(f"{region:5s} (+/-{half_width:.2f} m): {len(s)} states  "
              f"x [{c[:, 0].min():+.4f}, {c[:, 0].max():+.4f}]  "
              f"y [{c[:, 1].min():+.4f}, {c[:, 1].max():+.4f}]")

    out["num_steps_wait"] = np.asarray(NUM_STEPS_WAIT)
    out["table_height"] = np.asarray(float(env.model.mujoco_arena.table_offset[2]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"\nwrote {args.out}  (post-settle states, indices {EVAL_IDX_BASE}-{idx - 1})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

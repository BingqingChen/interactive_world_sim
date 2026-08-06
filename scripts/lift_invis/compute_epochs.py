"""Print the training config numbers that match a Lift condition to a target gradient-step budget.

Two problems this solves.

**Equal budget.** The three conditions must get the SAME number of gradient steps, not the same
number of epochs: the 450-demo sets have ~9x more samples per epoch than the 50-demo set, so a
fixed epoch count would hand them ~9x the optimization purely because their epochs are longer,
confounding data quantity with training length.

**Saving the final epoch.** The DP workspace only checkpoints when `epoch % checkpoint_every == 0`.
With `checkpoint_every: 50` and 294 epochs (indices 0-293) the last save is epoch 250 and the final
state is never written -- which silently turns "evaluate the final checkpoint" into "evaluate
whatever epoch happened to be a multiple of 50". So this prints `checkpoint_every == E` and
`num_epochs == E + 1`, which makes the LAST epoch index exactly E and therefore a save point.

`global_step` advances once per batch plus once per epoch (the workspace increments it in both
places), so the budget is matched on measured global_step, not on batches alone.

Usage:
    .venv/bin/python scripts/lift_invis/compute_epochs.py datasets/lift_vis_dp.zarr [target_steps]
"""

from __future__ import annotations

import math
import sys

DP_ROOT = "/home/jacobhb/projects/worth_doing/diffusion_policy"
sys.path.insert(0, DP_ROOT)

from diffusion_policy.dataset.lift_image_dataset import LiftImageDataset  # noqa: E402

# The centre run's evaluated checkpoint: epoch 250, global_step 17067 (measured from the payload).
DEFAULT_TARGET = 17_067
BATCH_SIZE = 64
HORIZON, N_OBS, N_ACT = 16, 2, 8


def main() -> int:
    zarr_path = sys.argv[1]
    target = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TARGET

    ds = LiftImageDataset(
        zarr_path,
        horizon=HORIZON,
        pad_before=N_OBS - 1,
        pad_after=N_ACT - 1,
        seed=42,
        val_ratio=0.05,
    )
    n_train_ep = int(ds.train_mask.sum())
    n_val_ep = int((~ds.train_mask).sum())
    n_samples = len(ds)
    batches = math.ceil(n_samples / BATCH_SIZE)

    # OFF-BY-ONE, measured the hard way: the checkpoint written during epoch index E's eval block
    # comes AFTER epoch E's training loop, so it reflects E+1 completed epochs, not E. Predicting
    # 24 * (710 + 1) = 17064 gave an actual 17749 (= 25 * ~710) and left the two treatment arms
    # 4.0% above the centre baseline's budget.
    per_epoch = batches
    n_epochs_needed = max(1, round(target / per_epoch))  # completed epochs to reach target
    last_idx = n_epochs_needed - 1                       # epoch INDEX of that checkpoint

    print(f"{zarr_path}")
    print(f"  episodes: {n_train_ep} train + {n_val_ep} val")
    print(f"  train samples: {n_samples}   batches/epoch @ bs{BATCH_SIZE}: {batches}")
    print(f"  target global_step: {target}")
    print(f"  -> num_epochs: {n_epochs_needed}        # epoch indices 0..{last_idx}")
    print(f"  -> checkpoint_every: {last_idx}      # so epoch index {last_idx} IS saved "
          f"(the last one)")
    print(f"  final checkpoint global_step: ~{n_epochs_needed * per_epoch} "
          f"({100 * n_epochs_needed * per_epoch / target:.1f}% of target)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

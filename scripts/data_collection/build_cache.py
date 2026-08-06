"""Build a dataset directory's cache.zarr.zip without the huge-RAM full conversion.

The stock conversion (`_convert_real_to_dp_replay`) accumulates every raw frame
in memory and then np.concatenate's them (~2x the raw dataset size — ~270 GB for
the combined 11k-episode mujoco+rotate-t train split). This script produces the
IDENTICAL cache by converting the episodes in batches (peak RAM ~ 2x one batch)
and stitching the batches together with a compressed-chunk copy.

Usage (run once per split, login node is fine):
    python scripts/data_collection/build_cache.py \
        --data_dir ${FILE_DIR}/data/interactive-world-sim-mujoco-data/train
    python scripts/data_collection/build_cache.py \
        --data_dir ${FILE_DIR}/data/interactive-world-sim-mujoco-data/val
"""
import argparse
import glob
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", required=True,
                    help="directory holding episode_*.hdf5 (a train/ or val/ split)")
    ap.add_argument("--config",
                    default=str(REPO_ROOT / "configurations/dataset/sim_aloha_dataset.yaml"),
                    help="dataset yaml providing shape_meta/action_mode")
    ap.add_argument("--batch", type=int, default=500, help="episodes per conversion batch")
    ap.add_argument("--force", action="store_true", help="overwrite an existing cache")
    args = ap.parse_args()

    import zarr
    from omegaconf import OmegaConf

    from interactive_world_sim.datasets.latent_dynamics.sim_aloha_dataset import (
        _merge_replay_buffers,
        load_replay_buffer,
    )

    data_dir = Path(args.data_dir).resolve()
    cache_path = data_dir / "cache.zarr.zip"
    if cache_path.exists():
        if not args.force:
            sys.exit(f"{cache_path} already exists — pass --force to rebuild")
        cache_path.unlink()

    cfg = OmegaConf.load(args.config)
    shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
    ctrl_mode = cfg.get("action_mode", "bimanual_push")

    ids = sorted(
        int(Path(p).stem.split("_")[-1])
        for p in glob.glob(str(data_dir / "episode_*.hdf5"))
    )
    if not ids:
        sys.exit(f"no episodes found in {data_dir}")
    batches = [ids[i : i + args.batch] for i in range(0, len(ids), args.batch)]
    print(f"{len(ids)} episodes in {len(batches)} batches of <= {args.batch}")

    buffers = []
    tmp_root = tempfile.mkdtemp(prefix="build_cache_")
    try:
        for bi, batch in enumerate(batches):
            bdir = Path(tmp_root) / f"batch_{bi}"
            bdir.mkdir()
            for i in batch:
                os.symlink(data_dir / f"episode_{i}.hdf5", bdir / f"episode_{i}.hdf5")
            print(f"batch {bi + 1}/{len(batches)}: episodes "
                  f"{batch[0]}..{batch[-1]} ({len(batch)})")
            buffers.append(
                load_replay_buffer(str(bdir), False, shape_meta, ctrl_mode=ctrl_mode)
            )
            shutil.rmtree(bdir)
        merged = buffers[0] if len(buffers) == 1 else _merge_replay_buffers(buffers)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"writing {cache_path} ({merged.n_episodes} episodes, {merged.n_steps} steps)")
    tmp = cache_path.with_suffix(".zip.tmp")
    if tmp.exists():
        tmp.unlink()
    with zarr.ZipStore(str(tmp)) as store:
        merged.save_to_store(store=store)
    os.replace(tmp, cache_path)
    print("Done.")


if __name__ == "__main__":
    main()

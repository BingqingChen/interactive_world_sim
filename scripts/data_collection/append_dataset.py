"""Append one IWS dataset's episodes into another's train/ and val/ folders.

Renumbers the source episodes to continue after the destination's highest
episode id, so the combined dataset lives in ONE root and trains through the
standard single-directory pipeline (no multi-root merging at load time).

Usage (cluster, after downloading the rotate-t data):
    python scripts/data_collection/append_dataset.py \
        --src ${FILE_DIR}/data/interactive-world-sim-rotate-t-data \
        --dst ${FILE_DIR}/data/interactive-world-sim-mujoco-data

Cache handling: an existing <dst>/<split>/cache.zarr.zip is STALE after new
episodes are added (the loader trusts it blindly and would silently ignore the
new data), so by default it is deleted and the next training run rebuilds it.
NOTE: a full rebuild of the combined mujoco+rotate-t cache holds every raw
frame in RAM twice (~270 GB peak) — run that first job on a high-memory node,
or pass --merge_cache to update the existing cache in place instead: the new
episodes are converted and appended to the old cache via compressed-chunk copy
(peak RAM ~ raw size of the NEW data only, ~40 GB here).
"""
import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parents[2]


def episode_ids(d: Path) -> list:
    return sorted(
        int(Path(p).stem.split("_")[-1])
        for p in glob.glob(str(d / "episode_*.hdf5"))
    )


def schema_of(path: Path) -> dict:
    out = {}
    with h5py.File(path, "r") as f:
        f.visititems(
            lambda name, o: out.update({name: (o.shape[1:], str(o.dtype))})
            if hasattr(o, "shape") and len(o.shape) > 0
            else None
        )
    return out


def merge_cache(src_split: Path, dst_split: Path, config_path: Path) -> bool:
    """Append src's episodes to dst's existing cache.zarr.zip (compressed-chunk
    copy; only src frames are jpeg2k-encoded). Returns True if updated."""
    import zarr
    from omegaconf import OmegaConf

    sys.path.insert(0, str(REPO_ROOT))
    from interactive_world_sim.datasets.latent_dynamics.sim_aloha_dataset import (
        _merge_replay_buffers,
        load_replay_buffer,
    )

    dst_cache = dst_split / "cache.zarr.zip"
    if not dst_cache.exists():
        print(f"  [{dst_split}] no existing cache — nothing to merge "
              "(next run builds the combined cache from scratch)")
        return False
    cfg = OmegaConf.load(config_path)
    shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
    ctrl_mode = cfg.get("action_mode", "bimanual_push")
    print(f"  [{dst_split}] loading destination cache ...")
    dst_buf = load_replay_buffer(str(dst_split), True, shape_meta, ctrl_mode)
    print(f"  [{src_split}] converting/loading source episodes ...")
    src_buf = load_replay_buffer(str(src_split), True, shape_meta, ctrl_mode)
    merged = _merge_replay_buffers([dst_buf, src_buf])
    tmp = dst_cache.with_suffix(".zip.tmp")
    print(f"  [{dst_split}] writing merged cache "
          f"({merged.n_episodes} episodes, {merged.n_steps} steps) ...")
    if tmp.exists():
        tmp.unlink()
    with zarr.ZipStore(str(tmp)) as store:
        merged.save_to_store(store=store)
    os.replace(tmp, dst_cache)
    src_cache = src_split / "cache.zarr.zip"
    if src_cache.exists():
        src_cache.unlink()  # spent: its episodes now live in dst
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="dataset root to take episodes from")
    ap.add_argument("--dst", required=True, help="dataset root to append into")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--copy", action="store_true", help="copy instead of move")
    ap.add_argument("--include_videos", action="store_true",
                    help="also bring <split>/videos/episode_N.mp4 along")
    ap.add_argument("--merge_cache", action="store_true",
                    help="update dst's existing cache.zarr.zip in place instead of "
                         "deleting it (avoids the huge full-rebuild on first training run)")
    ap.add_argument("--config", default=str(REPO_ROOT / "configurations/dataset/sim_aloha_dataset.yaml"),
                    help="dataset yaml providing shape_meta/action_mode (only used with --merge_cache)")
    ap.add_argument("--force", action="store_true", help="skip the schema-match check")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    src_root, dst_root = Path(args.src), Path(args.dst)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    transfer = shutil.copy2 if args.copy else shutil.move
    verb, verbed = ("copy", "copied") if args.copy else ("move", "moved")

    plans = []  # (split, [(src_path, dst_path, src_id, dst_id)])
    for split in splits:
        src_d, dst_d = src_root / split, dst_root / split
        if not src_d.is_dir() or not dst_d.is_dir():
            sys.exit(f"ERROR: missing split dir: {src_d if not src_d.is_dir() else dst_d}")
        src_ids, dst_ids = episode_ids(src_d), episode_ids(dst_d)
        if not src_ids:
            sys.exit(f"ERROR: no episodes in {src_d}")
        if not args.force:
            s, d = (
                schema_of(src_d / f"episode_{src_ids[0]}.hdf5"),
                schema_of(dst_d / f"episode_{dst_ids[0]}.hdf5"),
            )
            if s != d:
                sys.exit(
                    f"ERROR: schema mismatch in {split} (use --force to override):\n"
                    f"  src: {s}\n  dst: {d}"
                )
        next_id = (max(dst_ids) + 1) if dst_ids else 0
        moves = []
        for k, sid in enumerate(src_ids):
            moves.append(
                (src_d / f"episode_{sid}.hdf5", dst_d / f"episode_{next_id + k}.hdf5",
                 sid, next_id + k)
            )
        plans.append((split, moves))
        print(f"[{split}] dst has {len(dst_ids)} episodes (max id "
              f"{max(dst_ids) if dst_ids else '-'}); will {verb} {len(src_ids)} "
              f"as episode_{next_id}..episode_{next_id + len(src_ids) - 1}")

    if args.dry_run:
        print("dry run — nothing changed")
        return

    # Update caches BEFORE moving files (source conversion needs them in place).
    if args.merge_cache:
        for split, _ in plans:
            merge_cache(src_root / split, dst_root / split, Path(args.config))

    for split, moves in plans:
        for src_p, dst_p, _, _ in moves:
            assert not dst_p.exists(), f"collision: {dst_p}"
            transfer(str(src_p), str(dst_p))
            if args.include_videos:
                vid = src_p.parent / "videos" / (src_p.stem + ".mp4")
                if vid.exists():
                    vdir = dst_p.parent / "videos"
                    vdir.mkdir(exist_ok=True)
                    transfer(str(vid), str(vdir / (dst_p.stem + ".mp4")))
        print(f"[{split}] {verbed} {len(moves)} episodes")
        if not args.merge_cache:
            for stale in list(dst_root.glob(f"{split}/cache*.zarr.zip")) + list(
                dst_root.glob(f"{split}/cache*.zarr.zip.lock")
            ):
                stale.unlink()
                print(f"[{split}] deleted stale {stale.name} — the next training run "
                      "rebuilds it (see module docstring for the RAM implications)")

    print("Done.")


if __name__ == "__main__":
    main()

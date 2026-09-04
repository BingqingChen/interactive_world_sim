"""Scaling-law sweep v3 -- the §4 repeat with the strong expert.

Grid: R real x I imagined x 3 seeds. Both pools now come from the SAME
DP+residual expert (96.0% on region v3) over [-6,6]^2, so the scripted planner --
which fails ~84% of the time at x<-2 and left that band nearly empty in the
published pools -- is out of the experiment entirely.

Per cell: build its zarr on demand -> train -> assert the checkpoint really reached
the final epoch -> evaluate -> delete the zarr, ARCHIVE the checkpoints. Pre-building
all 48 cells costs ~90GB and has filled this disk once already, so the zarr is
transient; the checkpoints move to /data/storage rather than being deleted, so a cell
can still be re-evaluated later. See docs/data_filing_rules.md.

Ordering is seed-major: every cell of seed 1 first, so a partial run still gives a
complete grid at one seed rather than a fragment of all three.

Slices follow the published contract: conditions nest (r10 subset of r20 subset of
r50 ...) and imagined doses are prefixes of one pool. Seeds take DISJOINT real
slices; the imagined slice is [0:I] for every seed, because the pool holds 800 and
I goes to 800 -- the same design the published sweep used.

Usage:
  MUJOCO_GL=egl python scripts/run_scaling_v3_sweep.py            # full grid
  MUJOCO_GL=egl python scripts/run_scaling_v3_sweep.py --seeds 1  # one seed
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

IWS = Path(__file__).resolve().parent.parent
DP = Path("/home/jacobhb/projects/worth_doing/diffusion_policy")
PY = str(IWS / ".venv/bin/python")
MASK = str(IWS / "datasets/feasible_mask_v3.json")
OUT = IWS / "outputs/scaling_v3"
LOG = OUT / "logs"
SEED_OF = {1: 42, 2: 142, 3: 242}
MIN_FREE_GB = 100
ARCHIVE = Path("/data/storage/wm_archive/scaling_v3_checkpoints")


def free_gb():
    st = os.statvfs("/home")
    return st.f_bavail * st.f_frsize / 1e9


def sh(cmd, log, cwd, env_extra=None):
    env = dict(os.environ, MUJOCO_GL="egl", **(env_extra or {}))
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=cwd, env=env).returncode == 0


def ckpt_epoch(p):
    """Actual epoch inside a checkpoint -- never trust the filename or the log."""
    import dill
    import torch
    try:
        c = torch.load(p, map_location="cpu", pickle_module=dill, weights_only=False)
        return int(dill.loads(c["pickles"]["epoch"]))
    except Exception:
        return None


def run_cell(R, I, s, gpu, args):
    seed = SEED_OF[s]
    run = f"v3_r{R}i{I}_s{s}"
    rdir = DP / f"data/outputs/scaling_v3/{run}"
    ev = OUT / f"evals/{run}.json"
    if ev.exists():
        return f"EVAL-SKIP {run}"
    zarr = str(IWS / f"datasets/tmp_v3_{run}.zarr")

    if not (rdir / "checkpoints/latest.ckpt").exists():
        if free_gb() < MIN_FREE_GB:
            return f"ABORT {run}: only {free_gb():.0f}GB free (< {MIN_FREE_GB})"
        # seeds take disjoint real slices; imagined is a prefix shared by all seeds
        if not sh([PY, "scripts/data_collection/build_scaling_v3_zarrs.py",
                   "--real_slice", f"{(s-1)*R}:{s*R}", "--imag_slice", f"0:{I}",
                   "--out", zarr], LOG / f"build_{run}.log", IWS):
            return f"BUILD-FAILED {run}"
        ok = sh([PY, "train.py",
                 "--config-name=train_diffusion_unet_hybrid_rotate_t_scaling_v3",
                 "logging.mode=offline", "training.device=cuda:0",
                 f"training.seed={seed}", f"task.dataset.zarr_path={zarr}",
                 f"hydra.run.dir=data/outputs/scaling_v3/{run}"],
                LOG / f"train_{run}.log", DP, {"CUDA_VISIBLE_DEVICES": str(gpu)})
        shutil.rmtree(zarr, ignore_errors=True)          # transient by design
        Path(zarr + ".manifest.json").unlink(missing_ok=True)
        if not ok:
            return f"TRAIN-FAILED {run}"

    ck = rdir / "checkpoints/latest.ckpt"
    if not ck.exists():
        return f"NO-CKPT {run}"
    # The bug that invalidated two previous sweeps was invisible in the logs; verify
    # the artefact itself.
    e = ckpt_epoch(str(ck))
    if e is not None and e != args.epochs - 1:
        return f"BAD-EPOCH {run}: checkpoint is epoch {e}, expected {args.epochs-1}"

    ev.parent.mkdir(parents=True, exist_ok=True)
    # Record the first n_videos episodes. Because the init loop is seeded per episode
    # (`np.random.seed(seed + ep)`), episode k is the SAME initial state in every cell,
    # so these clips are directly comparable across the grid. Per-cell video_dir --
    # the eval default is one shared path and all 48 cells would overwrite each other.
    vdir = OUT / "rollouts" / run
    if not sh([PY, "scripts/eval_dp_rotate_t.py", "--ckpt", str(ck),
               "--n_episodes", str(args.eval_n), "--seed", "9000",
               "--max_steps", "800", "--n_videos", str(args.n_videos),
               "--video_dir", str(vdir),
               "--x_min", "-0.06", "--x_max", "0.06",
               "--y_min", "-0.06", "--y_max", "0.06",
               "--feasible_mask", MASK, "--metrics_json", str(ev)],
              LOG / f"eval_{run}.log", IWS, {"CUDA_VISIBLE_DEVICES": str(gpu)}):
        return f"EVAL-FAILED {run}"
    sr = json.loads(ev.read_text())["summary"]["success_rate"]
    # ARCHIVE the checkpoints off the NVMe now that the cell is evaluated. They must
    # not stay on / -- 48 cells x ~9GB is ~430GB and that exhausted the disk at cell 28
    # on the first attempt -- but they are not deleted either: without them a cell can
    # never be re-evaluated under a different protocol. (The first v3 run deleted them;
    # those models are gone.) rsync --remove-source-files unlinks the source only after
    # a verified transfer, and if the archive is unreachable the checkpoints are LEFT
    # IN PLACE rather than lost.
    src = rdir / "checkpoints"
    if src.is_dir():
        dest = ARCHIVE / run
        try:
            dest.mkdir(parents=True, exist_ok=True)
            rc = subprocess.run(["rsync", "-a", "--remove-source-files",
                                 f"{src}/", f"{dest}/"]).returncode
            if rc == 0:
                shutil.rmtree(src, ignore_errors=True)   # now-empty dirs
            else:
                return f"DONE {run}: {sr*100:.1f}%  (epoch {e})  ARCHIVE-FAILED rc={rc}"
        except OSError as err:
            return f"DONE {run}: {sr*100:.1f}%  (epoch {e})  ARCHIVE-UNAVAILABLE {err}"
    return f"DONE {run}: {sr*100:.1f}%  (epoch {e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, nargs="+", default=[10, 20, 50, 200])
    ap.add_argument("--I", type=int, nargs="+", default=[0, 50, 400, 800])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--eval_n", type=int, default=50)
    # ~0.15 MB/clip, so 10/cell is ~72 MB for the whole grid -- against 4-11 GPU-hours
    # to re-run the evals later just to look at a failure. Record by default.
    ap.add_argument("--n_videos", type=int, default=10,
                    help="episodes to record per cell (the same inits in every cell)")
    ap.add_argument("--epochs", type=int, default=21)
    args = ap.parse_args()
    (OUT / "evals").mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)

    cells = [(R, I, s) for s in args.seeds for R in args.R for I in args.I]
    print(f"{len(cells)} cells, seed-major, {free_gb():.0f}GB free", flush=True)
    t0 = time.time()
    # two at a time, one per GPU
    pending = list(cells)
    while pending:
        batch, pending = pending[:2], pending[2:]
        procs = []
        for gpu, (R, I, s) in enumerate(batch):
            procs.append((R, I, s, gpu))
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=2) as exe:
            futs = [exe.submit(run_cell, R, I, s, gpu, args) for R, I, s, gpu in procs]
            for f in futs:
                print(f"  [{(time.time()-t0)/3600:5.2f}h] {f.result()}", flush=True)
        if free_gb() < MIN_FREE_GB:
            print(f"ABORTING: {free_gb():.0f}GB free", flush=True)
            break

    rows = {}
    for f in (OUT / "evals").glob("v3_r*i*_s*.json"):
        import re
        m = re.match(r"v3_r(\d+)i(\d+)_s(\d)", f.stem)
        if m:
            rows.setdefault((int(m.group(1)), int(m.group(2))), {})[int(m.group(3))] = \
                json.loads(f.read_text())["summary"]["success_rate"] * 100
    print("\n=== scaling v3 (success %, mean +- SEM over seeds) ===")
    print("real \\ imagined  " + "".join(f"{I:>14d}" for I in args.I))
    import statistics as st
    for R in args.R:
        line = f"{R:>15d}  "
        for I in args.I:
            v = list(rows.get((R, I), {}).values())
            if not v:
                line += f"{'--':>14}"
            elif len(v) == 1:
                line += f"{v[0]:>10.1f}    "
            else:
                line += f"{st.mean(v):>8.1f}+-{st.stdev(v)/len(v)**0.5:<4.1f}"
        print(line, flush=True)
    (OUT / "grid.json").write_text(json.dumps(
        {f"{R},{I}": rows.get((R, I), {}) for R in args.R for I in args.I}, indent=1))


if __name__ == "__main__":
    main()

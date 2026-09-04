"""Autonomous filtered-self-imitation loop that trains a near-perfect expert on
region v3 ([-6,6]^2).

Each round: collect successful rollouts from the current best expert -> rebuild the
pool (region-v3 scripted demos + all self-imitation so far) -> retrain from scratch
-> evaluate -> promote the winner and repeat. Retraining from scratch on the whole
aggregate (rather than fine-tuning on the newest batch) is the DAgger prescription
and avoids the drift that incremental fine-tuning accumulates.

Why self-imitation rather than PPO: the scripted planner -- the only source of real
demos -- scores ~3/19 in the x<-2 band, so BC on scripted data can never cover the
region. The learned expert already scores ~8/12 there, so its own successes are the
missing supervision. Filtering by task success is a binary advantage filter needing
no action log-probs, which a diffusion policy cannot provide.

RUN ONE ROUND AT A TIME. Rounds are not a fixed schedule: the next strategy is
chosen from the round's actual result (which cells still fail, whether the harvest
is still yielding), so the normal invocation is `--rounds N` for a single N. The
target defaults to 1.01 -- deliberately unreachable -- because the goal is 100%,
and stopping at "good enough" would leave the last points unexamined.

Use `--epochs 40` to validate a strategy cheaply before paying for a full
120-epoch run; only commit the full budget once the short run shows a signal.

Guards that still apply within a multi-round invocation: a >5 pt regression, or two
rounds gaining <2 pts (self-imitation saturated -- change strategy).

Every stage is resumable: existing zarrs, checkpoints and eval JSONs are reused, so
the pipeline can be killed and relaunched without losing work.

Usage:
  MUJOCO_GL=egl python scripts/run_expertv3_pipeline.py --rounds 1
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

IWS = Path(__file__).resolve().parent.parent
DP = Path("/home/jacobhb/projects/worth_doing/diffusion_policy")
PY = str(IWS / ".venv/bin/python")
MASK = str(IWS / "datasets/feasible_mask_v3.json")
OUT = IWS / "outputs/expert_v3"
LOG = OUT / "logs"

# Round-1 seed checkpoint. MEASURED on region v3 (n=100 / n=50, seed 9000):
#   ev2_clean100_s42 ep70 : 72.0%   (in-training score 0.867)
#   ev2_s142         ep100: 80.0%   (in-training score 0.767)
# The two are statistically tied (~1.1 SE apart) and the in-training ranking is
# REVERSED relative to the real eval -- those scores came from mask v1 at n=30 and
# are not comparable. Round 1 was collected with clean_ep70; do not treat either as
# "the better expert" without an n>=100 region-v3 eval.
SEED0_CKPT = str(DP / "data/outputs/expertv2/ev2_clean100_s42/checkpoints"
                    "/epoch=0070-test_mean_score=0.867.ckpt")


def sh(cmd, log, cwd, env_extra=None):
    """Run a stage, streaming to a log file. Returns True on exit 0."""
    import os
    env = dict(os.environ, MUJOCO_GL="egl", **(env_extra or {}))
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=cwd, env=env).returncode == 0


def candidate_ckpts(run_dir):
    """Saved checkpoints, best in-training score first, then latest.ckpt.

    The in-training score is only n=30 rollouts (+-9 pts) -- too noisy to pick the
    final model on, and it already misled us once: expert-v2's `clean_ep70` scored
    0.867 in-training but 72% on the real n=100 region-v3 eval, while `ev2_s142`
    scored 0.767 in-training and 80%. So this returns CANDIDATES and the caller
    settles the ordering with a proper eval.
    """
    ck = sorted(glob.glob(str(Path(run_dir) / "checkpoints/epoch=*.ckpt")),
                key=lambda p: float(p.split("test_mean_score=")[1][:-5]), reverse=True)
    last = Path(run_dir) / "checkpoints/latest.ckpt"
    if last.exists():
        ck.append(str(last))
    return ck


def evaluate(ckpt, tag, n=100, seed=9000):
    """Region-v3 eval; cached by tag. Returns success rate in [0,1] or None."""
    js = OUT / f"eval_{tag}.json"
    if not js.exists():
        ok = sh([PY, "scripts/eval_dp_rotate_t.py", "--ckpt", ckpt,
                 "--n_episodes", str(n), "--seed", str(seed), "--max_steps", "800",
                 "--n_videos", "0", "--x_min", "-0.06", "--x_max", "0.06",
                 "--y_min", "-0.06", "--y_max", "0.06", "--feasible_mask", MASK,
                 "--metrics_json", str(js)], LOG / f"eval_{tag}.log", IWS)
        if not ok or not js.exists():
            return None
    return json.loads(js.read_text())["summary"]["success_rate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3])
    # NOTE: run_selfimit_round.sh's 3rd arg is a SCALE MULTIPLIER on each worker's
    # built-in attempt budget (strv 300 / neg 260 / ful 200), not an absolute count.
    # Passing an absolute count here would multiply the budget by that number.
    ap.add_argument("--scale", type=int, default=1,
                    help="multiplier on each worker's attempt budget (1 = ~1820 total)")
    # Default 1.01 = unreachable, i.e. never stop early on "good enough". The goal
    # is 100%; stopping at 95% would leave the last 5 points unexamined.
    ap.add_argument("--target", type=float, default=1.01)
    ap.add_argument("--eval_n", type=int, default=100)
    ap.add_argument("--n_candidates", type=int, default=2,
                    help="how many saved checkpoints to evaluate properly before\n                         picking the round's model (topk keeps 2)")
    ap.add_argument("--cap_per_cell", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override training.num_epochs -- use a short run (e.g. 40) to\n                         cheaply validate whether a strategy helps before paying for\n                         the full 120-epoch run")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ckpt, best, hist, stalls = SEED0_CKPT, 0.0, [], 0
    collect_proc = collect_fh = None
    collect_for = None  # which round the background collection is filling
    for r in args.rounds:
        t0 = time.time()
        print(f"\n=== ROUND {r} === collector: {Path(ckpt).name}", flush=True)

        # --- collect ---------------------------------------------------------
        if collect_proc is not None and collect_for == r:
            # This round's data is already being gathered in the background,
            # overlapped with the previous round's training. Wait for it -- the
            # periodic flush means partial zarrs exist on disk, so testing for
            # their presence would otherwise train on a half-written round.
            print(f"  waiting for overlapped round-{r} collection...", flush=True)
            rc = collect_proc.wait()
            collect_fh.close()
            collect_proc = collect_fh = collect_for = None
            if rc != 0:
                print(f"ROUND {r}: overlapped collection exit={rc}", flush=True)
        elif not list((IWS / f"datasets/selfimit/r{r}").glob("*.zarr")):
            if not sh(["bash", "scripts/run_selfimit_round.sh", str(r), ckpt,
                       str(args.scale)], LOG / f"round{r}_collect.log", IWS):
                print(f"ROUND {r}: collection FAILED", flush=True)
                break
        kept = sum(json.loads(Path(f).read_text())["n_kept"]
                   for f in glob.glob(str(IWS / f"datasets/selfimit/r{r}/*.manifest.json")))
        print(f"  collected {kept} successful rollouts", flush=True)

        # --- rebuild pool over ALL rounds so far -----------------------------
        rounds_so_far = [x for x in args.rounds if x <= r]
        if not sh([PY, "scripts/data_collection/build_expertv3_pool.py",
                   "--rounds", *map(str, rounds_so_far),
                   "--cap_per_cell", str(args.cap_per_cell)],
                  LOG / f"round{r}_pool.log", IWS):
            print(f"ROUND {r}: pool build FAILED", flush=True)
            break
        pman = Path(str(IWS / "datasets/rotate_t_expertv3_pool.zarr") + ".manifest.json")
        pool = json.loads(pman.read_text())
        # each round overwrites the pool zarr and its manifest; keep a per-round copy
        # so a finished round stays reproducible
        (OUT / f"pool_manifest_r{r}.json").write_text(pman.read_text())
        print(f"  pool: {pool['n_episodes']} eps "
              f"({pool['n_real']} real + {pool['n_selfim']} self-imitation), "
              f"{pool['cells_covered']}/{pool['cells_total']} cells", flush=True)

        # --- overlap: start the NEXT round's collection on GPU 1 -------------
        # Training owns GPU 0 for ~9h while GPU 1 would otherwise idle. The data
        # collected here comes from the current expert rather than the one being
        # trained, so it is slightly off-policy for DAgger purposes -- but it is
        # still success-filtered expert data in the cells that need it, and the
        # alternative is leaving half the hardware idle for a third of the run.
        nxt = next((x for x in args.rounds if x > r), None)
        if nxt is not None and not list((IWS / f"datasets/selfimit/r{nxt}").glob("*.zarr")):
            nlog = LOG / f"round{nxt}_collect.log"
            nlog.parent.mkdir(parents=True, exist_ok=True)
            collect_fh = open(nlog, "w")
            collect_proc = subprocess.Popen(
                ["bash", "scripts/run_selfimit_round.sh", str(nxt), ckpt,
                 str(args.scale)],
                stdout=collect_fh, stderr=subprocess.STDOUT, cwd=IWS,
                env=dict(os.environ, MUJOCO_GL="egl", GPUS="1"))
            collect_for = nxt
            print(f"  overlapped: round {nxt} collection started on GPU 1 "
                  f"(collector {Path(ckpt).name})", flush=True)

        # --- train from scratch on the aggregate -----------------------------
        run = f"expertv3_r{r}"
        rdir = DP / f"data/outputs/expert_v3/{run}"
        if not (rdir / "checkpoints/latest.ckpt").exists():
            if not sh([PY, "train.py",
                       "--config-name=train_diffusion_unet_hybrid_rotate_t_expertv3",
                       "logging.mode=offline", "training.device=cuda:0",
                       "training.seed=42",
                       *( [f"training.num_epochs={args.epochs}"] if args.epochs else [] ),
                       f"hydra.run.dir=data/outputs/expert_v3/{run}"],
                      LOG / f"round{r}_train.log", DP,
                      {"CUDA_VISIBLE_DEVICES": "0"}):
                print(f"ROUND {r}: training FAILED", flush=True)
                break
        # --- select the checkpoint on a REAL eval, not the n=30 in-training score
        cands = candidate_ckpts(rdir)[:args.n_candidates]
        if not cands:
            print(f"ROUND {r}: no checkpoint produced", flush=True)
            break
        scored = []
        for c in cands:
            v = evaluate(c, f"{run}_{Path(c).stem[:14]}", n=args.eval_n)
            if v is not None:
                print(f"  {Path(c).name}: {v*100:.1f}%", flush=True)
                scored.append((v, c))
        if not scored:
            print(f"ROUND {r}: eval FAILED for every candidate", flush=True)
            break
        scored.sort()
        sr, ck = scored[-1]
        if len(scored) > 1:
            print(f"  selected {Path(ck).name} ({sr*100:.1f}%); "
                  f"spread across candidates "
                  f"{(scored[-1][0]-scored[0][0])*100:.1f} pts", flush=True)
        hist.append((r, sr, kept, pool["n_episodes"]))
        print(f"ROUND {r}: {sr*100:.1f}% on region v3 "
              f"(best so far {best*100:.1f}%), {(time.time()-t0)/3600:.1f}h", flush=True)
        (OUT / "pipeline_history.json").write_text(json.dumps(
            [dict(round=a, success=b, collected=c, pool=d) for a, b, c, d in hist],
            indent=1))

        # --- promote / early-stop -------------------------------------------
        if sr >= args.target:
            print(f"TARGET REACHED at round {r}: {sr*100:.1f}% >= "
                  f"{args.target*100:.0f}%", flush=True)
            ckpt = ck
            break
        if sr < best - 0.05:
            print(f"REGRESSION ({sr*100:.1f}% vs best {best*100:.1f}%) -- "
                  f"stopping, keeping the earlier expert", flush=True)
            break
        stalls = stalls + 1 if sr < best + 0.02 else 0
        if sr > best:
            best, ckpt = sr, ck
        if stalls >= 2:
            print("STALLED: two rounds with <2pt gain -- self-imitation has "
                  "saturated, a different strategy is needed", flush=True)
            break

    # An early stop (target/regression/stall) can leave the overlapped collection
    # for a round that will never run. Kill it rather than leaving 7 workers
    # occupying a GPU indefinitely.
    if collect_proc is not None and collect_proc.poll() is None:
        print(f"  stopping orphaned round-{collect_for} collection", flush=True)
        collect_proc.terminate()
        try:
            collect_proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            collect_proc.kill()
        # the round script backgrounds its workers, so they outlive it
        subprocess.run(["pkill", "-f", "collect_policy_rollouts.py"], check=False)
    if collect_fh is not None:
        collect_fh.close()

    print("\n=== PIPELINE SUMMARY ===", flush=True)
    for a, b, c, d in hist:
        print(f"  round {a}: {b*100:5.1f}%  (+{c} rollouts, pool {d})", flush=True)
    print(f"BEST EXPERT: {best*100:.1f}%  {ckpt}", flush=True)
    (OUT / "best_expert.json").write_text(json.dumps(
        dict(ckpt=ckpt, success_rate=best, region="feasible_mask_v3.json",
             history=[dict(round=a, success=b) for a, b, _, _ in hist]), indent=1))


if __name__ == "__main__":
    main()

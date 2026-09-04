"""DAgger-in-world-model chain for the region-v2 scaling experiment.

For one (R, seed): start from a BC policy trained on R real demos (the I=0 cell,
shared with the off-policy grid), then repeat for K rounds:

  1. roll the CURRENT student out inside the world model, 100 episodes, from
     region-v2 inits, open-loop (the sim is touched only for frame 0 -- no
     resyncing, so this is a deployable procedure);
  2. label every hallucinated observation with the EXPERT (expert-v2) at that
     same hallucinated observation -- states ~ student, labels ~ expert;
  3. aggregate D_k = R real + all imagined rounds so far;
  4. retrain FROM SCRATCH on D_k at full budget -> the I=100k condition.

Round k's snapshot IS the I=100k cell, so a single 8-round chain yields
I in {100,200,400,800} without recomputing shared prefixes. Retraining from
scratch (rather than warm-starting) follows Ross et al. and avoids carrying
optimizer/LR-scheduler state across rounds -- the failure mode that silently
degraded two earlier runs here.

Usage:
  MUJOCO_GL=egl python scripts/run_dagger_v2.py --R 20 --seed 42 --gpu 0
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

IWS = Path(__file__).resolve().parent.parent
DP = Path("/home/jacobhb/projects/worth_doing/diffusion_policy")
PY = str(IWS / ".venv/bin/python")
EXPERT = str(DP / "data/outputs/expertv2/ev2_s142/checkpoints"
                 "/epoch=0100-test_mean_score=0.767.ckpt")
MASK = str(IWS / "datasets/feasible_mask_v2.json")
REGION = ["--x_min", "-0.02", "--x_max", "0.06", "--y_min", "-0.06", "--y_max", "0.04"]
sys.path.insert(0, str(DP))
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


def run(cmd, log, gpu, cwd):
    env = {"MUJOCO_GL": "egl", "CUDA_VISIBLE_DEVICES": str(gpu),
           "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=f, cwd=cwd, env=env).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--per_round", type=int, default=100)
    ap.add_argument("--ep_len", type=int, default=400)
    args = ap.parse_args()

    tag = f"r{args.R}_s{args.seed}"
    work = IWS / "datasets/dagger_v2" / tag
    logs = IWS / "outputs/dagger_v2/logs"
    work.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # round 0 student = the I=0 BC policy from the off-policy grid (same object)
    student = DP / f"data/outputs/scaling_v2/v2_r{args.R}i0_s{ {42:1,142:2,242:3}[args.seed] }/checkpoints/latest.ckpt"
    if not student.exists():
        print(f"FATAL: round-0 policy missing: {student}")
        return 1
    print(f"[{tag}] round-0 student = {student}", flush=True)

    real = ReplayBuffer.copy_from_path(
        str(IWS / "datasets/rotate_t_regionv2_real_dp.zarr"))
    round_zarrs = []
    for k in range(1, args.rounds + 1):
        I = k * args.per_round
        rollout_z = work / f"round{k}_raw.zarr"
        agg_z = work / f"agg_i{I}.zarr"
        run_name = f"dag_r{args.R}i{I}_s{args.seed}"
        run_dir = DP / "data/outputs/dagger_v2" / run_name
        ck = run_dir / "checkpoints/latest.ckpt"
        if ck.exists():
            print(f"[{tag}] round {k} (I={I}) SKIP, ckpt exists", flush=True)
            student = ck
            round_zarrs.append(rollout_z)
            continue

        # ---- 1+2: student rolls out in the WM; expert labels the same obs ----
        if not rollout_z.exists():
            rc = run([PY, "-u", str(IWS / "scripts/collect_imagined_rotate_t.py"),
                      "--dp_ckpt", str(student), "--labeler_ckpt", EXPERT,
                      "--n_episodes", str(args.per_round), "--ep_len", str(args.ep_len),
                      "--batch", "25", "--random_init", *REGION,
                      "--feasible_mask", MASK, "--terminal_mode", "angle",
                      "--min_len", "40", "--keep_failures",
                      "--env_seed_base", str(800000 + 1000 * args.R + 100 * args.seed + k),
                      "--seed", str(args.seed + k),
                      "--out_zarr", str(rollout_z), "--video_dir", ""],
                     logs / f"{run_name}_collect.log", args.gpu, IWS)
            if rc != 0:
                print(f"[{tag}] COLLECT-FAILED round {k}"); return 1
        round_zarrs.append(rollout_z)
        print(f"[{tag}] round {k}: collected {args.per_round} eps "
              f"({time.time()-t0:.0f}s)", flush=True)

        # ---- 3: aggregate R real + every round so far ----
        rb = ReplayBuffer.create_empty_numpy()
        for i in range(args.R):
            rb.add_episode(real.get_episode(i))
        for rz in round_zarrs:
            src = ReplayBuffer.copy_from_path(str(rz))
            for i in range(src.n_episodes):
                rb.add_episode(src.get_episode(i))
            del src
        rb.save_to_path(str(agg_z), if_exists="replace")
        print(f"[{tag}] round {k}: aggregate {rb.n_episodes} eps "
              f"({rb.n_steps} steps) -> {agg_z.name}", flush=True)

        # ---- 4: retrain from scratch at full budget ----
        rc = run([PY, "train.py",
                  "--config-name=train_diffusion_unet_hybrid_rotate_t_scaling_v2",
                  "logging.mode=offline", "training.device=cuda:0",
                  f"training.seed={args.seed}", "training.resume=False",
                  f"task.dataset.zarr_path={agg_z}",
                  f"hydra.run.dir=data/outputs/dagger_v2/{run_name}"],
                 logs / f"train_{run_name}.log", args.gpu, DP)
        if not ck.exists():
            print(f"[{tag}] TRAIN-FAILED round {k} (rc={rc})"); return 1
        student = ck
        print(f"[{tag}] ROUND-DONE {run_name} ({time.time()-t0:.0f}s)", flush=True)

    (IWS / f"outputs/dagger_v2/{tag}_manifest.json").write_text(json.dumps(dict(
        R=args.R, seed=args.seed, rounds=args.rounds, per_round=args.per_round,
        expert=EXPERT, region=MASK, wall_s=int(time.time() - t0)), indent=1))
    print(f"[{tag}] CHAIN COMPLETE in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

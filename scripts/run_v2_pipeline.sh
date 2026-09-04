#!/bin/bash
# Master orchestrator for the region-v2 experiment.
#   1. wait for the off-policy imagined pools (v2A/v2B) to finish
#   2. build the 20 (R,I) grid zarrs
#   3. run the off-policy scaling sweep (60 runs, seed-major)
#   4. run the 12 DAgger chains (2 at a time, one per GPU)
#   5. evaluate every DAgger snapshot
# Everything is resumable: each stage skips work whose output already exists.
set -u
IWS=/home/jacobhb/projects/worth_doing/interactive_world_sim
DP=/home/jacobhb/projects/worth_doing/diffusion_policy
PY=$IWS/.venv/bin/python
MASK=$IWS/datasets/feasible_mask_v2.json
EVALS=$IWS/outputs/dagger_v2/evals
mkdir -p "$EVALS" "$IWS/outputs/dagger_v2/logs"

echo "=== STAGE 1: waiting for imagined pools ==="
while pgrep -f collect_imagined_rotate_t.*imagined_v2 > /dev/null 2>&1 || \
      pgrep -f "out_zarr.*rotate_t_imagined_v2" > /dev/null 2>&1; do sleep 60; done
for sh in A B; do
  [ -d "$IWS/datasets/rotate_t_imagined_v2$sh.zarr" ] || { echo "POOL-MISSING $sh"; exit 1; }
done
echo "POOLS READY"

echo "=== STAGE 2: building grid zarrs ==="
(cd "$IWS" && "$PY" scripts/data_collection/build_scaling_v2_zarrs.py --grid) \
  > "$IWS/outputs/scaling_v2/build_grid.log" 2>&1 \
  && echo "GRID BUILT" || { echo "GRID-BUILD-FAILED"; exit 1; }

echo "=== STAGE 3: off-policy scaling sweep ==="
bash "$IWS/scripts/run_scaling_v2_sweep.sh"
echo "STAGE 3 COMPLETE"

echo "=== STAGE 4: DAgger chains ==="
gpu=0
for s in 42 142 242; do
  for R in 10 20 50 200; do
    (cd "$IWS" && MUJOCO_GL=egl "$PY" -u scripts/run_dagger_v2.py \
      --R "$R" --seed "$s" --gpu "$gpu" \
      > "$IWS/outputs/dagger_v2/logs/chain_r${R}_s${s}.log" 2>&1 \
      && echo "CHAIN-DONE r${R}_s${s}" || echo "CHAIN-FAILED r${R}_s${s}") &
    gpu=$(( (gpu + 1) % 2 ))
    [ $gpu -eq 0 ] && wait
  done
done
wait
echo "STAGE 4 COMPLETE"

echo "=== STAGE 5: DAgger evals ==="
n=0
for s in 42 142 242; do
  for R in 10 20 50 200; do
    for I in 100 200 400 800; do
      run=dag_r${R}i${I}_s${s}
      [ -f "$EVALS/$run.json" ] && { echo "EVAL-SKIP $run"; continue; }
      ck=$DP/data/outputs/dagger_v2/$run/checkpoints/latest.ckpt
      [ -f "$ck" ] || { echo "EVAL-FAILED $run (no ckpt)"; continue; }
      (cd "$IWS" && MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$((n % 2)) "$PY" scripts/eval_dp_rotate_t.py \
        --ckpt "$ck" --n_episodes 50 --seed 9000 --max_steps 800 --n_videos 0 \
        --x_min -0.02 --x_max 0.06 --y_min -0.06 --y_max 0.04 \
        --feasible_mask "$MASK" --metrics_json "$EVALS/$run.json" \
        > "$IWS/outputs/dagger_v2/logs/eval_$run.log" 2>&1 \
        && echo "EVAL-DONE $run: $("$PY" -c "import json;print(round(json.load(open('$EVALS/$run.json'))['summary']['success_rate']*100,1))" 2>/dev/null)%" \
        || echo "EVAL-FAILED $run") &
      n=$((n + 1))
      [ $((n % 6)) -eq 0 ] && wait
    done
  done
done
wait
echo "V2 PIPELINE COMPLETE"

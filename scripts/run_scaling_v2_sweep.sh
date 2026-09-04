#!/bin/bash
# Region-v2 scaling grid: R x I, off-policy imagined data, seed-major ordering
# (seed 1 of every cell first, so partial results are usable).
#
# R in {10,20,50,200} real demos (regionv2 pool, cell-interleaved prefix)
# I in {0,100,200,400,800} imagined episodes (expert-v2 in WM, region-v2 inits)
# 3 seeds -> 60 runs. Train + eval under feasible region v2.
set -u
IWS=/home/jacobhb/projects/worth_doing/interactive_world_sim
DP=/home/jacobhb/projects/worth_doing/diffusion_policy
PY=$IWS/.venv/bin/python
MASK=$IWS/datasets/feasible_mask_v2.json
EVALS=$IWS/outputs/scaling_v2/evals
mkdir -p "$EVALS" "$IWS/outputs/scaling_v2/logs"

R_VALUES="10 20 50 200"
I_VALUES="0 100 200 400 800"
declare -A SEED=([1]=42 [2]=142 [3]=242)

for s in 1 2 3; do
  # ---- train every cell for this seed (2 at a time, one per GPU) ----
  gpu=0
  for R in $R_VALUES; do
    for I in $I_VALUES; do
      run=v2_r${R}i${I}_s${s}
      dir=$DP/data/outputs/scaling_v2/$run
      zarr=$IWS/datasets/tmp_v2_r${R}i${I}.zarr
      if [ -f "$dir/checkpoints/latest.ckpt" ]; then echo "TRAIN-SKIP $run"; continue; fi
      [ -d "$zarr" ] || { echo "TRAIN-FAILED $run (missing $zarr)"; continue; }
      (cd "$DP" && MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$gpu "$PY" train.py \
        --config-name=train_diffusion_unet_hybrid_rotate_t_scaling_v2 \
        logging.mode=offline training.device=cuda:0 training.seed="${SEED[$s]}" \
        task.dataset.zarr_path="$zarr" \
        hydra.run.dir="data/outputs/scaling_v2/$run" \
        > "$IWS/outputs/scaling_v2/logs/train_$run.log" 2>&1
       if [ -f "$dir/checkpoints/latest.ckpt" ]; then echo "TRAIN-DONE $run"
       else echo "TRAIN-FAILED $run"; fi) &
      gpu=$(( (gpu + 1) % 2 ))
      [ $gpu -eq 0 ] && wait
    done
  done
  wait
  echo "SEED $s TRAINING COMPLETE"

  # ---- eval every cell for this seed (6 concurrent, 3 per GPU) ----
  n=0
  for R in $R_VALUES; do
    for I in $I_VALUES; do
      run=v2_r${R}i${I}_s${s}
      [ -f "$EVALS/$run.json" ] && { echo "EVAL-SKIP $run"; continue; }
      ck=$DP/data/outputs/scaling_v2/$run/checkpoints/latest.ckpt
      [ -f "$ck" ] || { echo "EVAL-FAILED $run (no ckpt)"; continue; }
      (cd "$IWS" && MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$((n % 2)) "$PY" scripts/eval_dp_rotate_t.py \
        --ckpt "$ck" --n_episodes 50 --seed 9000 --max_steps 800 --n_videos 0 \
        --x_min -0.02 --x_max 0.06 --y_min -0.06 --y_max 0.04 \
        --feasible_mask "$MASK" --metrics_json "$EVALS/$run.json" \
        > "$IWS/outputs/scaling_v2/logs/eval_$run.log" 2>&1 \
        && echo "EVAL-DONE $run: $("$PY" -c "import json;print(round(json.load(open('$EVALS/$run.json'))['summary']['success_rate']*100,1))" 2>/dev/null)%" \
        || echo "EVAL-FAILED $run") &
      n=$((n + 1))
      [ $((n % 6)) -eq 0 ] && wait
    done
  done
  wait
  echo "SEED $s COMPLETE"
done
echo "SCALING V2 SWEEP COMPLETE"

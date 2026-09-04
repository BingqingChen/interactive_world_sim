#!/bin/bash
# Sweep driver for the expert-sim-relabeling ablation (r20+i400).
# Per seed: train both arms sequentially, then eval both concurrently --
# ordered so seed-1 results for BOTH arms land first.
# Usage: MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 bash scripts/run_relabel_mix_sweep.sh
set -u
IWS=/home/jacobhb/projects/worth_doing/interactive_world_sim
DP=/home/jacobhb/projects/worth_doing/diffusion_policy
PY=$IWS/.venv/bin/python
EVALS=$IWS/outputs/relabel_mix/evals
mkdir -p "$EVALS" "$IWS/outputs/relabel_mix/videos"

declare -A ZARR=(
  [pair]=$IWS/datasets/rotate_t_r20_pair400_dp.zarr
  [relab]=$IWS/datasets/rotate_t_r20_relabel400_dp.zarr
  [retr]=$IWS/datasets/rotate_t_r20_retr400_dp.zarr
  [retrv]=$IWS/datasets/rotate_t_r20_retrv400_dp.zarr
)
ARMS="pair relab retr retrv"
declare -A SEED=([1]=42 [2]=142 [3]=242)

for s in 1 2 3; do
  for arm in $ARMS; do
    run=sc_r20${arm}400_s${s}
    dir=$DP/data/outputs/scaling/$run
    if [ -f "$dir/checkpoints/latest.ckpt" ]; then
      echo "TRAIN-SKIP $run (checkpoint exists)"; continue
    fi
    echo "TRAIN-START $run seed=${SEED[$s]}"
    # NOTE: with rollout_every=99 the topk checkpointer crashes at the epoch-5
    # checkpoint (KeyError: test_mean_score). The ORIGINAL published sweep hit
    # the same crash -- every published sc_* latest.ckpt is epoch 5 / step 6005.
    # We deliberately reproduce that de-facto protocol: a run counts as trained
    # if latest.ckpt exists after train.py exits (it is saved before the crash).
    (cd "$DP" && MUJOCO_GL=egl "$PY" train.py \
      --config-name=train_diffusion_unet_hybrid_rotate_t_scaling \
      logging.mode=offline training.device=cuda:0 training.num_epochs=21 \
      training.seed="${SEED[$s]}" training.rollout_every=99 \
      task.dataset.zarr_path="${ZARR[$arm]}" \
      hydra.run.dir="data/outputs/scaling/$run" \
      > "$IWS/outputs/relabel_mix/train_$run.log" 2>&1)
    if [ -f "$dir/checkpoints/latest.ckpt" ]; then
      echo "TRAIN-DONE $run (de-facto epoch-5 ckpt, matching published sweep)"
    else
      echo "TRAIN-FAILED $run"; exit 1
    fi
  done
  for arm in $ARMS; do
    run=sc_r20${arm}400_s${s}
    [ -f "$EVALS/$run.json" ] && { echo "EVAL-SKIP $run"; continue; }
    (cd "$IWS" && MUJOCO_GL=egl "$PY" scripts/eval_dp_rotate_t.py \
      --ckpt "$DP/data/outputs/scaling/$run/checkpoints/latest.ckpt" \
      --n_episodes 50 --seed 7000 --max_steps 800 \
      --metrics_json "$EVALS/$run.json" \
      --video_dir "$IWS/outputs/relabel_mix/videos/$run" --n_videos 2 \
      > "$IWS/outputs/relabel_mix/eval_$run.log" 2>&1 \
      && echo "EVAL-DONE $run: $("$PY" -c "import json;d=json.load(open('$EVALS/$run.json'));print(d)" 2>/dev/null | head -c 200)" \
      || echo "EVAL-FAILED $run") &
  done
  wait
  echo "SEED $s COMPLETE"
done
echo "RELABEL SWEEP COMPLETE"

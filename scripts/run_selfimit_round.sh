#!/bin/bash
# One round of filtered self-imitation over region v3 ([-6,6]^2).
#
# Fans out 7 workers that roll the current expert out in the real sim and keep only
# the successful episodes (success predicate identical to eval_dp_rotate_t.py).
#
# Budget is skewed hard toward x<-2, where the scripted planner cannot produce demos
# at all (measured 3/19) and the real pool is consequently starved -- the x=-6 column
# holds 3 demos across 6 cells, while every column at x>=-2 holds 24-44 per cell. The
# learned expert scores ~8/12 in that same band, so this is where self-imitation adds
# supervision that no amount of scripted collection could.
#
#   strv* (3 workers x 300)  x in [-6,-2]  the 11 starved cells
#   neg*  (2 workers x 260)  x in [-6, 0]  starved band + its boundary
#   ful*  (2 workers x 200)  x in [-6, 6]  full region, guards against forgetting
#
# One worker per group uses --action_noise 0.003 for extra state diversity.
#
# Usage: run_selfimit_round.sh <round> <ckpt> [scale]   (scale multiplies attempts)
#
# Set GPUS to restrict which devices the workers use, e.g. GPUS="1" to keep all
# collection on GPU 1 while a training job owns GPU 0 (round-(N+1) collection
# overlapped with round-N training).
set -u
IWS=/home/jacobhb/projects/worth_doing/interactive_world_sim
PY=$IWS/.venv/bin/python
RND=$1; CKPT=$2; SCALE=${3:-1}
OUT=$IWS/datasets/selfimit/r$RND
LOG=$IWS/outputs/expert_v3/logs
mkdir -p "$OUT" "$LOG"
read -r -a GPU_LIST <<< "${GPUS:-0 1}"
gi=0

# name x_min x_max attempts seed_offset noise   (GPU assigned round-robin)
run() {
  local nm=$1 xlo=$2 xhi=$3 n=$4 so=$5 nz=$6
  local gpu=${GPU_LIST[$(( gi % ${#GPU_LIST[@]} ))]}
  gi=$(( gi + 1 ))
  n=$(( n * SCALE ))
  if [ -d "$OUT/$nm.zarr" ]; then echo "SKIP $nm"; return; fi
  (cd "$IWS" && MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$gpu "$PY" -u scripts/collect_policy_rollouts.py \
     --ckpt "$CKPT" --n_attempts "$n" --max_steps 800 --action_noise "$nz" \
     --x_min "$xlo" --x_max "$xhi" --y_min -0.06 --y_max 0.06 \
     --feasible_mask "$IWS/datasets/feasible_mask_v3.json" \
     --seed $(( RND * 100000 + so )) --out_zarr "$OUT/$nm.zarr" \
     > "$LOG/selfimit_r${RND}_$nm.log" 2>&1 \
   && echo "DONE $nm $(grep -o 'kept [0-9]*/[0-9]*' "$LOG/selfimit_r${RND}_$nm.log" | tail -1)" \
   || echo "FAILED $nm") &
}

run strvA -0.06 -0.02 300 11000 0.0
run strvB -0.06 -0.02 300 12000 0.003
run strvC -0.06 -0.02 300 13000 0.0
run negA  -0.06  0.00 260 21000 0.0
run negB  -0.06  0.00 260 22000 0.003
run fulA  -0.06  0.06 200 31000 0.0
run fulB  -0.06  0.06 200 32000 0.003
wait
echo "SELFIMIT ROUND $RND COMPLETE"
"$PY" - <<EOF
import glob, json
tot = a = 0
for f in sorted(glob.glob("$OUT/*.manifest.json")):
    m = json.load(open(f)); tot += m["n_kept"]; a += m["n_attempts_done"]
    print(f"  {f.split('/')[-1][:-14]:6s} {m['n_kept']:4d}/{m['n_attempts_done']:4d}  ({m['accept_rate']*100:.0f}%)")
print(f"ROUND $RND: {tot} successful episodes from {a} attempts ({tot/max(a,1)*100:.0f}%)")
EOF

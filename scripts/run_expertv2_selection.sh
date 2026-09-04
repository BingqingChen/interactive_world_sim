#!/bin/bash
# Expert-v2 checkpoint selection: top-3 ckpts/seed (by in-training score) evaluated
# n=50 on the SELECTION seed (8000, never the report seed), feasible-region protocol.
# Then evaluate the OLD expert + the selection winner(s) on the REPORT seed (9000).
set -u
IWS=/home/jacobhb/projects/worth_doing/interactive_world_sim
DP=/home/jacobhb/projects/worth_doing/diffusion_policy
PY=$IWS/.venv/bin/python
MASK=$IWS/datasets/feasible_mask_v1.json
SEL=$IWS/outputs/expertv2/select
REP=$IWS/outputs/expertv2/report
mkdir -p "$SEL" "$REP"

OLD_EXPERT="$DP/data/outputs/2026.07.24/00.23.18_train_diffusion_unet_hybrid_rotate_t_image_eval/checkpoints/epoch=0010-test_mean_score=0.417.ckpt"

run_eval () {
  local ckpt="$1" name="$2" seed="$3" gpu="$4" outdir="$5"
  [ -f "$outdir/$name.json" ] && { echo "SEL-SKIP $name"; return; }
  (cd "$IWS" && MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/eval_dp_rotate_t.py \
    --ckpt "$ckpt" --n_episodes 50 --seed "$seed" --max_steps 800 --n_videos 0 \
    --feasible_mask "$MASK" --metrics_json "$outdir/$name.json" \
    > "$outdir/${name}.log" 2>&1 \
    && echo "EVAL-DONE $name: $("$PY" -c "import json;d=json.load(open('$outdir/$name.json'));print(round(d['summary']['success_rate']*100,1))" 2>/dev/null)%" \
    || echo "EVAL-FAILED $name")
}

echo "=== SELECTION PASS (seed 8000, feasible mask) ==="
i=0
while read -r ckpt; do
  [ -z "$ckpt" ] && continue
  gpu=$((i % 2)); i=$((i+1))
  name="$(basename "$(dirname "$(dirname "$ckpt")")")_$(basename "$ckpt" .ckpt)"
  run_eval "$ckpt" "$name" 8000 $gpu "$SEL" &
done < <(tr ' ' '\n' < "$SEL/ckpts_s42.txt")
while read -r ckpt; do
  [ -z "$ckpt" ] && continue
  gpu=$((i % 2)); i=$((i+1))
  name="$(basename "$(dirname "$(dirname "$ckpt")")")_$(basename "$ckpt" .ckpt)"
  run_eval "$ckpt" "$name" 8000 $gpu "$SEL" &
done < <(tr ' ' '\n' < "$SEL/ckpts_s142.txt")
wait
echo "SELECTION PASS COMPLETE"

# pick winner: highest success_rate across all selection JSONs
best=$("$PY" - <<'EOF'
import json, glob
best = max(glob.glob("outputs/expertv2/select/*.json"),
           key=lambda f: json.load(open(f))["summary"]["success_rate"])
print(best)
EOF
)
winner_ckpt=$("$PY" -c "
import json
d = json.load(open('$best'))
print(d['summary']['ckpt'])")
echo "WINNER: $best -> $winner_ckpt"
echo "$winner_ckpt" > "$SEL/winner_ckpt.txt"

echo "=== REPORT PASS (seed 9000, feasible mask): old expert vs winner ==="
run_eval "$OLD_EXPERT" "old_expert" 9000 0 "$REP" &
run_eval "$winner_ckpt" "expertv2_winner" 9000 1 "$REP" &
wait
echo "REPORT PASS COMPLETE"

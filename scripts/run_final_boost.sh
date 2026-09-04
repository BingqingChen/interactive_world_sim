#!/bin/bash
# Full expert pipeline: collected data -> strong diffusion policy -> residual PPO boost.
#
#   STAGE 1  wait for round-1 self-imitation collection to finish
#   STAGE 2  build the pool (all 1187 real demos + every harvested rollout) and train
#            the diffusion policy on it, selecting the checkpoint by a real n=100
#            region-v3 eval rather than the noisy n=30 in-training score
#   STAGE 3  residual PPO on top of that DP
#   STAGE 4  evaluate the PPO snapshots and keep the best
#
# Why the snapshots matter: in the v2 PPO run the residual peaked around iteration
# 40-45 -- deterministic paired eval (n=50, CRN): 96.0% vs the base's 74.0%,
# discordant 11-0, McNemar p=0.00098 -- and had fully DEGRADED by iteration 80,
# scoring 74.0%, i.e. no better than the unmodified base. Taking the final iterate,
# or trusting the on-policy trace, would have thrown the entire +22 pts away.
set -u
IWS=/home/jacobhb/projects/worth_doing/interactive_world_sim
DP=/home/jacobhb/projects/worth_doing/diffusion_policy
PY=$IWS/.venv/bin/python
OUT=$IWS/outputs/final_expert
LOG=$OUT/logs
mkdir -p "$OUT" "$LOG"

echo "=== STAGE 1: waiting for collection ==="
while pgrep -f 'collect_policy_rollouts\.py' > /dev/null; do sleep 60; done
$PY - <<'EOF'
import glob, json
t=a=0
for f in sorted(glob.glob('/home/jacobhb/projects/worth_doing/interactive_world_sim/datasets/selfimit/r1/*.manifest.json')):
    m=json.load(open(f)); t+=m['n_kept']; a+=m['n_attempts_done']
    print(f"  {f.split('/')[-1][:-14]:6s} {m['n_kept']:4d}/{m['n_attempts_done']:4d}")
print(f"COLLECTED {t} successful rollouts from {a} attempts ({t/max(a,1)*100:.0f}%)")
EOF

echo "=== STAGE 2: pool + diffusion policy ==="
cd "$IWS"
MUJOCO_GL=egl $PY -u scripts/run_expertv3_pipeline.py \
  --rounds 1 --target 1.01 --eval_n 100 --scale 1 --cap_per_cell 0 \
  > "$LOG/stage2_dp.log" 2>&1
echo "stage 2 exit=$?"
BEST=$($PY -c "
import json,sys
try:
    d=json.load(open('$IWS/outputs/expert_v3/best_expert.json'))
    print(d['ckpt'] if d.get('ckpt') else '')
except Exception: print('')")
if [ -z "$BEST" ] || [ ! -f "$BEST" ]; then
  echo "STAGE 2 FAILED: no diffusion-policy checkpoint produced; stopping."
  exit 1
fi
echo "DP checkpoint: $BEST"
$PY -c "
import json;d=json.load(open('$IWS/outputs/expert_v3/best_expert.json'))
print(f\"  region-v3 eval: {d['success_rate']*100:.1f}%\")" 2>/dev/null

echo "=== STAGE 3: residual PPO on the new DP ==="
# Settings that produced the working iter-40 residual. Long run (400 iters) with
# EARLY STOPPING on degradation plus best-snapshot retention, rather than a fixed
# short cap: v2 peaked near iter 45 and then lost the stall-breaking behaviour, so
# the rule keeps the peak and abandons the run once it has clearly fallen behind.
mkdir -p "$OUT/ppo"
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 $PY -u scripts/ppo_residual_rotate_t.py \
  --ckpt "$BEST" --n_envs 12 --iters 400 --chunks_per_iter 16 \
  --epochs 10 --minibatch 128 --lr 1e-3 --delta_max 0.015 --log_std0 -1.0 \
  --snapshot_every 10 --early_stop_drop 0.05 --early_stop_patience 12 \
  --early_stop_after 30 --feasible_mask "$IWS/datasets/feasible_mask_v3.json" \
  --out "$OUT/ppo" > "$LOG/stage3_ppo.log" 2>&1
echo "stage 3 exit=$?"

echo "=== STAGE 4: pick the best residual snapshot by real eval ==="
# A 400-iteration run snapshots up to 40 times; evaluating all of them at 50 paired
# episodes each would take days. Shortlist instead: the best-by-on-policy snapshot
# plus the top few, then let the DETERMINISTIC eval decide among them. On-policy
# score is only trusted to shortlist -- never to pick.
CANDS=$($PY - <<EOF
import json, glob, os, re
out="$OUT"
h=json.load(open(f"{out}/ppo/history.json"))
have={int(re.search(r"residual_iter(\d+)\.pt$", f).group(1))
      for f in glob.glob(f"{out}/ppo/residual_iter*.pt")}
rows=[(r["success_rate_last100"], r["iter"]) for r in h
      if r["iter"] in have and r["success_rate_last100"]==r["success_rate_last100"]]
rows.sort(reverse=True)
picks=[f"{out}/ppo/residual_iter{it}.pt" for _,it in rows[:3]]
if os.path.exists(f"{out}/ppo/residual_best.pt"):
    picks.append(f"{out}/ppo/residual_best.pt")
print(" ".join(dict.fromkeys(picks)))
EOF
)
echo "candidates: $CANDS"
for snap in $CANDS; do
  [ -f "$snap" ] || continue
  tag=$(basename "$snap" .pt)
  [ -f "$OUT/eval_$tag.json" ] && { echo "  skip $tag"; continue; }
  MUJOCO_GL=egl $PY -u scripts/eval_residual_policy.py \
    --residual "$snap" --n_episodes 50 --with_base \
    --feasible_mask "$IWS/datasets/feasible_mask_v3.json" \
    --metrics_json "$OUT/eval_$tag.json" > "$LOG/eval_$tag.log" 2>&1
  $PY -c "
import json;d=json.load(open('$OUT/eval_$tag.json'))['summary']
print(f\"  $tag: {d['success_rate']*100:.1f}%  (base {d.get('base_success_rate',0)*100:.1f}%, \"
      f\"discordant +{d.get('residual_wins',0)}/-{d.get('base_wins',0)})\")" 2>/dev/null
done

echo "=== FINAL ==="
$PY - <<EOF
import glob, json
rows=[]
for f in glob.glob("$OUT/eval_residual_*.json"):
    d=json.load(open(f))['summary']
    rows.append((d['success_rate'], f.split('/')[-1][5:-5], d))
rows.sort(reverse=True)
for sr,tag,d in rows:
    print(f"  {tag:22s} {sr*100:5.1f}%  base {d.get('base_success_rate',0)*100:.1f}%  "
          f"discordant +{d.get('residual_wins',0)}/-{d.get('base_wins',0)}")
if rows:
    sr,tag,d=rows[0]
    print(f"\nBEST EXPERT: {tag} at {sr*100:.1f}% on region v3")
    json.dump(dict(snapshot=tag, success_rate=sr, base=d.get('base_success_rate'),
                   dp_ckpt="$BEST"), open("$OUT/final_expert.json","w"), indent=1)
EOF
echo "FINAL BOOST PIPELINE COMPLETE"

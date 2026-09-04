#!/bin/bash
# Wait for the imagined pool, report both pools, then run the 48-cell sweep.
IWS=/home/jacobhb/projects/worth_doing/interactive_world_sim
PY=$IWS/.venv/bin/python
cd "$IWS"
echo "=== waiting for imagined pool ==="
while pgrep -f 'collect_imagined_rotate_t\.py' > /dev/null; do sleep 60; done
echo "=== pools ready $(date) ==="
$PY - <<'PYEOF'
import glob, sys
import numpy as np
sys.path.insert(0,'/home/jacobhb/projects/worth_doing/diffusion_policy')
from diffusion_policy.common.replay_buffer import ReplayBuffer
for name, pat in (("real","datasets/scaling_v3/real/*.zarr"),
                  ("imagined","datasets/scaling_v3/imag/*.zarr")):
    tot=steps=0; lens=[]
    for z in sorted(glob.glob(pat)):
        rb=ReplayBuffer.copy_from_path(z); tot+=rb.n_episodes; steps+=rb.n_steps
        lens += [len(rb.get_episode(i)['action']) for i in range(rb.n_episodes)]
    if lens:
        l=np.array(lens)
        print(f"  {name:9s} {tot:4d} episodes  {steps:7d} frames  "
              f"len min/med/mean/max = {l.min()}/{int(np.median(l))}/{l.mean():.0f}/{l.max()}")
PYEOF
echo "=== launching sweep (seed-major, 2 cells at a time) ==="
exec env MUJOCO_GL=egl $PY -u scripts/run_scaling_v3_sweep.py --eval_n 50

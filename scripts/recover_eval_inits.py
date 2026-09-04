"""Recover the per-episode init positions of an eval that did not record them, and
join them onto its results so the success rate can be decomposed by region.

eval_dp_rotate_t.py's init loop is fully deterministic given --seed and the protocol:
`np.random.seed(seed+ep)`, then `env.reset(seed=seed+trial)` with trial advanced only
by mask rejections, which depend on physics alone -- never on the policy. So replaying
just the reset/settle loop (no policy, no rollout) reproduces exactly the inits any
eval used, and episode i here is episode i there.

That makes an 80%-overall number answerable in the way that matters: WHICH cells the
failures are in. Cells that fail because supervision is missing are fixable by
self-imitation; cells where nothing ever succeeds need real exploration instead.

Usage:
  MUJOCO_GL=egl python scripts/recover_eval_inits.py \
      --eval_json outputs/expert_v3/baseline_expertv2_v3.json \
      --n_episodes 50 --seed 9000 --feasible_mask datasets/feasible_mask_v3.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "data_collection"))
import eval_dp_rotate_t as E  # noqa: E402
import collect_rotate_t as C  # noqa: E402
import gym_aloha.env as gae  # noqa: E402
from gym_aloha.env import AlohaEnv  # noqa: E402
from yixuan_utilities.kinematics_helper import KinHelper  # noqa: E402
from interactive_world_sim.utils.pose_utils import PoseType, pose_convert  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json", required=True)
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--x_min", type=float, default=-0.06)
    ap.add_argument("--x_max", type=float, default=0.06)
    ap.add_argument("--y_min", type=float, default=-0.06)
    ap.add_argument("--y_max", type=float, default=0.06)
    ap.add_argument("--feasible_mask", required=True)
    args = ap.parse_args()

    m = json.loads(Path(args.feasible_mask).read_text())
    mask, edges = np.array(m["mask"], bool), np.array(m["grid_cm"]["edges"])
    env = AlohaEnv("pusht")
    kin = KinHelper(robot_name="trossen_vx300s")
    gae.sample_pusht_pose = C.make_upright_pose_sampler(
        (args.x_min, args.x_max), (args.y_min, args.y_max))

    def cell_and_ok():
        s = env._env.task.get_observation(env._env.physics)["env_state"]
        x, y = s[0] * 100, s[1] * 100
        if not (edges[0] <= x < edges[-1] and edges[0] <= y < edges[-1]):
            return None, (x, y), False
        i = int(np.digitize(x, edges) - 1)
        j = int(np.digitize(y, edges) - 1)
        return (i, j), (x, y), bool(mask[i, j])

    inits, trial = [], 0
    for ep in range(args.n_episodes):
        np.random.seed(args.seed + ep)
        while True:
            env.reset(seed=args.seed + trial); trial += 1
            C.stabilize_t(env)
            o = env._env.task.get_observation(env._env.physics)
            lb = pose_convert(o["left_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            rb = pose_convert(o["right_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            C.settle_arms(env, np.stack([lb, rb]), kin, np.zeros(6),
                          E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM)
            c, xy, ok = cell_and_ok()
            if abs(C.t_angle(env)) <= C.UPRIGHT_TOL and ok:
                break
        inits.append((c, xy))
        if (ep + 1) % 10 == 0:
            print(f"  recovered {ep+1}/{args.n_episodes}", flush=True)

    d = json.loads(Path(args.eval_json).read_text())
    eps = d["episodes"]
    if len(eps) != len(inits):
        print(f"WARNING: eval has {len(eps)} episodes, recovered {len(inits)} -- "
              f"check --n_episodes/--seed match the eval's protocol")
    for r, (c, xy) in zip(eps, inits):
        r["init_xy_cm"] = [round(xy[0], 2), round(xy[1], 2)]
        r["init_cell"] = list(c) if c else None
    Path(args.eval_json).write_text(json.dumps(d, indent=1))

    per = defaultdict(lambda: [0, 0])
    for r in eps:
        if r["init_cell"]:
            k = tuple(r["init_cell"])
            per[k][1] += 1; per[k][0] += int(r["success"])
    print(f"\n{Path(args.eval_json).name}: "
          f"{d['summary']['success_rate']*100:.1f}% overall\n")
    print("success by x column (cm):")
    col = defaultdict(lambda: [0, 0])
    for (i, j), (s, n) in per.items():
        col[i][0] += s; col[i][1] += n
    for i in sorted(col):
        s, n = col[i]
        print(f"  x={edges[i]:+3.0f}: {s:3d}/{n:3d}  {s/n*100:5.1f}%")
    print("\ncells with any failure:")
    for (i, j), (s, n) in sorted(per.items()):
        if s < n:
            print(f"  ({edges[i]:+3.0f},{edges[j]:+3.0f}) {s}/{n}")


if __name__ == "__main__":
    main()

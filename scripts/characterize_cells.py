"""Per-cell characterization of a policy over the feasible-mask-v1 grid.

Stage 2 of the feasible-region-v2 plan. The pooled 400-episode estimate that
motivated mask v2 is heterogeneous (mixed checkpoints; n ranges 2..30 per cell),
so cells like (-4,+4) n=2 cannot support an in/out decision. This runs a uniform
measurement: for each of the 40 mask-v1 cells, evaluate ONE policy on N inits
drawn from that cell.

Containment is enforced two ways so a measured cell really is that cell:
  * the pose sampler is pinned to the cell via --x_min/--x_max/--y_min/--y_max
  * a generated single-cell mask is passed as --feasible_mask, so any init that
    settle-drifts out of the cell (or out of the grid) is rejected and resampled

Outputs outputs/expertv2/cell_char.json: per-cell n, successes, rate, Wilson CI,
plus the per-episode records for later re-analysis.

Usage:
  MUJOCO_GL=egl python scripts/characterize_cells.py --ckpt <ckpt> [--n 20]
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

IWS = Path(__file__).resolve().parent.parent


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mask", default=str(IWS / "datasets/feasible_mask_v1.json"))
    ap.add_argument("--n", type=int, default=20,
                    help="TARGET episodes per cell (existing prior episodes count "
                         "toward it; only the deficit is collected)")
    ap.add_argument("--prior", default=None,
                    help="JSON of prior per-cell counts {'i,j': [successes, n]} to "
                         "credit against the target (see pool_prior_evals.py)")
    ap.add_argument("--seed", type=int, default=11000,
                    help="disjoint from 7000/8000/9000 used elsewhere")
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--n_gpus", type=int, default=2)
    ap.add_argument("--per_gpu", type=int, default=3)
    ap.add_argument("--out_dir", default=str(IWS / "outputs/expertv2/cell_char"))
    args = ap.parse_args()

    m = json.loads(Path(args.mask).read_text())
    edges = np.array(m["grid_cm"]["edges"])
    B = m["grid_cm"]["n"]
    mask = np.array(m["mask"], bool)
    out = Path(args.out_dir)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    cells = [(i, j) for i in range(B) for j in range(B) if mask[i, j]]
    prior = json.loads(Path(args.prior).read_text()) if args.prior else {}

    # one single-cell mask file per cell (enforces post-settle containment)
    jobs = []
    for (i, j) in cells:
        have = prior.get(f"{i},{j}", [0, 0])[1]
        need = max(0, args.n - have)
        if need == 0:
            continue
        single = np.zeros((B, B), bool)
        single[i, j] = True
        mp = out / "masks" / f"cell_{i}_{j}.json"
        mp.write_text(json.dumps(dict(
            grid_cm=m["grid_cm"], mask=[[bool(v) for v in row] for row in single],
            note=f"single-cell mask for characterization of cell ({i},{j})")))
        jobs.append((i, j, mp, need))
    print(f"{len(cells)} in-mask cells, target n={args.n}/cell; "
          f"{len(jobs)} cells need top-up, {sum(j[3] for j in jobs)} new episodes "
          f"(prior credits {sum(v[1] for v in prior.values())} existing)")

    t0 = time.time()
    slots = args.n_gpus * args.per_gpu
    running = []
    done = 0
    for k, (i, j, mp, need) in enumerate(jobs):
        res = out / f"cell_{i}_{j}.json"
        if res.exists():
            done += 1
            continue
        while len(running) >= slots:
            for p in running[:]:
                if p[0].poll() is not None:
                    running.remove(p)
                    done += 1
                    print(f"  [{done}/{len(jobs)}] cell {p[1]} done "
                          f"({time.time() - t0:.0f}s)", flush=True)
            if len(running) >= slots:
                time.sleep(5)
        gpu = len(running) % args.n_gpus
        cmd = [str(IWS / ".venv/bin/python"), str(IWS / "scripts/eval_dp_rotate_t.py"),
               "--ckpt", args.ckpt, "--n_episodes", str(need),
               "--seed", str(args.seed + 100 * k), "--max_steps", str(args.max_steps),
               "--n_videos", "0", "--feasible_mask", str(mp),
               "--metrics_json", str(res),
               "--x_min", f"{edges[i] / 100:.4f}", "--x_max", f"{edges[i + 1] / 100:.4f}",
               "--y_min", f"{edges[j] / 100:.4f}", "--y_max", f"{edges[j + 1] / 100:.4f}"]
        env = {"MUJOCO_GL": "egl", "CUDA_VISIBLE_DEVICES": str(gpu),
               "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
        log = open(out / f"cell_{i}_{j}.log", "w")
        running.append((subprocess.Popen(cmd, stdout=log, stderr=log, cwd=IWS, env=env),
                        f"({edges[i]:+.0f},{edges[j]:+.0f})"))
    for p in running:
        p[0].wait()
        done += 1
    print(f"all cells done in {time.time() - t0:.0f}s")

    # ---- aggregate ----
    rows = {}
    for (i, j) in cells:
        key = f"{i},{j}"
        k_, n = prior.get(key, [0, 0])
        res = out / f"cell_{i}_{j}.json"
        if res.exists():
            d = json.load(open(res))
            n += d["summary"]["n_episodes"]
            k_ += d["summary"]["n_success"]
        elif key not in prior:
            print(f"  MISSING cell {key} (no prior, no new eval)")
            continue
        lo, hi = wilson(k_, n)
        rows[key] = dict(
            i=i, j=j, x_cm=float(edges[i]), y_cm=float(edges[j]),
            n=n, successes=k_, rate=round(k_ / n, 4),
            wilson_lo=round(lo, 4), wilson_hi=round(hi, 4))
    summary = dict(
        ckpt=args.ckpt, mask_v1=args.mask, n_per_cell=args.n, seed=args.seed,
        n_cells=len(rows), cells=rows, prior=args.prior,
        pooled_rate=round(sum(r["successes"] for r in rows.values())
                          / max(sum(r["n"] for r in rows.values()), 1), 4),
        wall_s=int(time.time() - t0))
    (IWS / "outputs/expertv2/cell_char.json").write_text(json.dumps(summary, indent=1))
    print(f"\npooled over all in-mask cells: {summary['pooled_rate']*100:.1f}%")
    for key in sorted(rows, key=lambda k: rows[k]["rate"]):
        r = rows[key]
        print(f"  ({r['x_cm']:+.0f},{r['y_cm']:+.0f})  {r['rate']*100:5.1f}%  "
              f"({r['successes']:2d}/{r['n']})  95% CI [{r['wilson_lo']*100:.0f},"
              f"{r['wilson_hi']*100:.0f}]")


if __name__ == "__main__":
    main()

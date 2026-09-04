"""Turn a finished self-imitation round into cell targets for the next one.

Round 1 aims attempts by *region* (the -x half), which is a proxy. Once a round has
run we have something better: a measured per-cell accept rate and a measured per-cell
demo count. This picks the cells that are actually starved or actually hard, so the
next round spends its budget where the expert is still weak instead of re-confirming
the cells it already solves.

A cell is a target if EITHER
  * pool count < --min_demos            (not enough supervision yet), or
  * accept rate < --min_accept          (the expert still fails there)
ranked by need = shortfall in demos, weighted up when the accept rate is low.

Prints a ready-to-paste --cells argument list.

Usage: python scripts/plan_next_selfimit_round.py --round 1
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

IWS = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--min_demos", type=int, default=30)
    ap.add_argument("--min_accept", type=float, default=0.7)
    ap.add_argument("--max_cells", type=int, default=12)
    args = ap.parse_args()

    m = json.loads((IWS / "datasets/feasible_mask_v3.json").read_text())
    edges, mask = np.array(m["grid_cm"]["edges"]), np.array(m["mask"], bool)

    suc, att = defaultdict(int), defaultdict(int)
    for f in glob.glob(str(IWS / f"datasets/selfimit/r{args.round}/*.manifest.json")):
        for cs, (s, a) in json.loads(Path(f).read_text()).get("per_cell", {}).items():
            c = tuple(int(v) for v in cs.split(","))
            suc[c] += s; att[c] += a

    pool_man = Path(str(IWS / "datasets/rotate_t_expertv3_pool.zarr") + ".manifest.json")
    pool = defaultdict(int)
    if pool_man.exists():
        for cs, n in json.loads(pool_man.read_text())["per_cell_counts"].items():
            pool[tuple(int(v) for v in cs.split(","))] = n

    rows = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if not mask[i, j]:
                continue
            c = (i, j)
            a, s, n = att[c], suc[c], pool[c]
            rate = s / a if a else None
            short = max(0, args.min_demos - n)
            hard = rate is not None and rate < args.min_accept
            if short == 0 and not hard:
                continue
            # low accept rate means each attempt yields less, so ask for more
            need = short * (2.0 if hard else 1.0) + (30 if hard and short == 0 else 0)
            rows.append((need, c, n, s, a, rate))
    rows.sort(reverse=True)

    print(f"round {args.round}: {sum(att.values())} attempts, {sum(suc.values())} kept "
          f"({sum(suc.values())/max(sum(att.values()),1)*100:.0f}%)")
    print(f"{'cell (cm)':>12} {'pool':>5} {'round acc':>11} {'need':>6}")
    for need, c, n, s, a, rate in rows[:args.max_cells]:
        rs = f"{s}/{a} ({rate*100:.0f}%)" if a else "  -  "
        print(f"  ({edges[c[0]]:+3.0f},{edges[c[1]]:+3.0f}) {n:5d} {rs:>11} {need:6.0f}")
    if not rows:
        print("no cell is starved or hard -- coverage is uniform; "
              "next round should just add volume, or the strategy has saturated")
        return
    cells = " ".join(f"{c[0]},{c[1]}" for _, c, *_ in rows[:args.max_cells])
    print(f"\n--cells {cells}")
    (IWS / f"outputs/expert_v3/round{args.round + 1}_cells.txt").write_text(cells + "\n")


if __name__ == "__main__":
    main()

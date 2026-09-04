"""Pool the already-collected expert-v2-family eval episodes into per-cell counts,
so characterize_cells.py only has to collect the deficit rather than re-measure.

Sources: the 6 selection evals (seed 8000) + 2 report evals (seed 9000) =
400 episodes over 8 checkpoints of the expert-v2 family (all 70-80% overall).
Pooling across the family is deliberate: the mask is a TASK-REGION definition,
so "what this class of policy can do here" is a more robust basis than one
checkpoint's quirks.

Cell assignment is STRICT (out-of-grid episodes dropped, not clamped) -- the
16 out-of-grid episodes were the clamping-bug contamination, and dropping them
materially changes edge cells: (+6,-6) 40% -> 57%, (+6,+0) 43% -> 50%.

Init positions are reconstructed deterministically from the eval seeds; they are
cached in outputs/expertv2/eval_inits_8000_9000.json by reconstruct_eval_inits.py.

Writes outputs/expertv2/prior_cell_counts.json: {"i,j": [successes, n]}.

Usage: python scripts/pool_prior_evals.py
"""
import json
from pathlib import Path

import numpy as np

IWS = Path(__file__).resolve().parent.parent


def main():
    m = json.loads((IWS / "datasets/feasible_mask_v1.json").read_text())
    edges = np.array(m["grid_cm"]["edges"])
    inits = json.loads((IWS / "outputs/expertv2/eval_inits_8000_9000.json").read_text())
    pos = {s: {int(e): (x, y) for e, x, y in v} for s, v in inits.items()}

    def strict_cell(x, y):
        if not (edges[0] <= x < edges[-1] and edges[0] <= y < edges[-1]):
            return None
        return int(np.digitize(x, edges) - 1), int(np.digitize(y, edges) - 1)

    counts, dropped, n_files = {}, 0, 0
    for sub, seed in (("select", "8000"), ("report", "9000")):
        for f in sorted((IWS / "outputs/expertv2" / sub).glob("*.json")):
            n_files += 1
            for r in json.load(open(f))["episodes"]:
                c = strict_cell(*pos[seed][r["episode"]])
                if c is None:
                    dropped += 1
                    continue
                key = f"{c[0]},{c[1]}"
                s, n = counts.get(key, [0, 0])
                counts[key] = [s + int(r["success"]), n + 1]

    out = IWS / "outputs/expertv2/prior_cell_counts.json"
    out.write_text(json.dumps(counts, indent=1, sort_keys=True))
    tot = sum(v[1] for v in counts.values())
    suc = sum(v[0] for v in counts.values())
    print(f"pooled {n_files} eval files -> {tot} in-grid episodes over "
          f"{len(counts)} cells ({dropped} out-of-grid dropped)")
    print(f"pooled rate {suc / tot * 100:.1f}%  ->  {out}")


if __name__ == "__main__":
    main()

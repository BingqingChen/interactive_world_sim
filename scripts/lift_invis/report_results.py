"""Assemble the final 3-policies x 2-regions table from the per-condition eval JSONs.

Reports success rate +/- s.e.m. per cell, and the two contrasts the experiment exists to measure:

    invisible gain = centre+invisible - centre     (on the large region)
    visible gain   = centre+visible   - centre     (on the large region)

If the invisible gain is positive, the extra out-of-region demos helped even though they never
showed the cube -- i.e. the benefit came from motion/workspace coverage rather than from seeing
the object. The visible gain is the ceiling for comparison.

Differences of proportions on the SAME 50 initial states are paired, so the s.e.m. of a difference
is reported from the per-episode disagreement (McNemar-style), not from the two marginal s.e.m.s
added in quadrature -- the latter would overstate the uncertainty.

Usage:
    .venv/bin/python scripts/lift_invis/report_results.py outputs/lift_invis/results_*.json
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

LABEL_ORDER = ["centre", "vis", "invis"]
PRETTY = {"centre": "centre", "vis": "centre+visible", "invis": "centre+invisible"}
REGIONS = ["small", "large"]


def paired_diff(a: list[bool], b: list[bool]) -> tuple[float, float]:
    """(b - a) mean and its paired s.e.m. over the same episodes."""
    assert len(a) == len(b), "conditions were evaluated on different numbers of episodes"
    d = [int(y) - int(x) for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    return mean, math.sqrt(var / n)


def main() -> int:
    paths = sys.argv[1:]
    if len(paths) == 1 and any(c in paths[0] for c in "*?"):
        paths = sorted(glob.glob(paths[0]))
    if not paths:
        print("usage: report_results.py <results_*.json ...>")
        return 2

    res = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        res[d["label"]] = d

    labels = [x for x in LABEL_ORDER if x in res] + [x for x in res if x not in LABEL_ORDER]

    w = max(len(PRETTY.get(x, x)) for x in labels) + 2
    print(f"\n{'policy':<{w}}" + "".join(f"{r + ' region':>22}" for r in REGIONS))
    print("-" * (w + 22 * len(REGIONS)))
    for lab in labels:
        row = f"{PRETTY.get(lab, lab):<{w}}"
        for r in REGIONS:
            cell = res[lab]["regions"].get(r)
            if cell is None:
                row += f"{'--':>22}"
            else:
                row += f"{100 * cell['success_rate']:>13.1f}% +/-{100 * cell['sem']:>4.1f}"
        print(row)

    # The contrasts, on the large region (and small, for completeness).
    if "centre" in res:
        print()
        for r in REGIONS:
            base = res["centre"]["regions"].get(r)
            if base is None:
                continue
            for lab in ("vis", "invis"):
                cell = res.get(lab, {}).get("regions", {}).get(r)
                if cell is None:
                    continue
                m, se = paired_diff(base["success_flags"], cell["success_flags"])
                z = m / se if se else float("nan")
                print(f"{r:5s}  {PRETTY[lab]} - centre = {100 * m:+.1f}% +/- {100 * se:.1f}% "
                      f"(paired, n={len(base['success_flags'])}, {z:+.1f} s.e.)")

    # The single cleanest contrast: invisible vs visible differ ONLY in the cube's pixels --
    # identical actions, identical physics, identical initial states, identical budget. Any gap
    # here is attributable to seeing the object and to nothing else.
    if "vis" in res and "invis" in res:
        print()
        for r in REGIONS:
            a = res["vis"]["regions"].get(r)
            b = res["invis"]["regions"].get(r)
            if a is None or b is None:
                continue
            m, se = paired_diff(a["success_flags"], b["success_flags"])
            z = m / se if se else float("nan")
            print(f"{r:5s}  centre+invisible - centre+visible = {100 * m:+.1f}% "
                  f"+/- {100 * se:.1f}% (paired, n={len(a['success_flags'])}, {z:+.1f} s.e.)")

    print("\nnote: all cells use the FINAL checkpoint and the same fixed initial states.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

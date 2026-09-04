"""Rebuild the scaling-law payload for PairScope from whatever results exist now.

Separates two eval protocols so stale numbers can never be mistaken for current:
  v2      -- CORRECT protocol: init sampler pinned to region v2 bounds
             (x in [-0.02,0.06] m, y in [-0.06,0.04] m) + post-settle mask check,
             identical to what training rollouts and imagined/DAgger rollouts use.
  v2_old  -- SUPERSEDED: sampler left at +-8cm with only the post-settle mask
             check, so ~5% of accepted inits had drifted in from outside the
             region and the within-region density differed from training.

Run after any batch of evals lands, then re-export the viewer.

Usage: python scripts/refresh_scaling_viz.py
"""
import glob
import json
import re
from pathlib import Path

IWS = Path(__file__).resolve().parent.parent
DP = Path("/home/jacobhb/projects/worth_doing/diffusion_policy")


def collect(pattern, tag):
    out = {}
    for f in glob.glob(pattern):
        m = re.search(rf"{tag}_r(\d+)i(\d+)_s(\d)", f)
        if not m:
            continue
        try:
            d = json.load(open(f))["summary"]
        except Exception:
            continue
        out.setdefault(f"{m.group(1)},{m.group(2)}", {})[int(m.group(3))] = \
            round(d["success_rate"] * 100, 1)
    return out


def main():
    pub = json.loads((IWS / "outputs/scaling_v2/published_grid.json").read_text())
    payload = dict(
        published=pub["published"], pub_R=pub["R"], pub_I=pub["I"],
        v2=collect(str(IWS / "outputs/scaling_v2/evals/*.json"), "v2"),
        v2_old=collect(str(IWS / "outputs/scaling_v2/evals_oldprotocol/*.json"), "v2"),
        dagger=collect(str(IWS / "outputs/dagger_v2/evals/*.json"), "dag"),
        v2_R=[10, 20, 50, 200], v2_I=[0, 100, 200, 400, 800],
        v2_trained=len(glob.glob(str(DP / "data/outputs/scaling_v2/*/checkpoints/latest.ckpt"))),
        dagger_trained=len(glob.glob(str(DP / "data/outputs/dagger_v2/*/checkpoints/latest.ckpt"))),
        v2_total_runs=60, dagger_total_runs=48,
        region_v2=dict(cells=20, area_pct=31, measured=91.1,
                       bounds="x in [-2,6] cm, y in [-6,4] cm"),
        protocol=("inits sampled uniformly from the region bounds, arms settled, "
                  "resampled if the post-settle T left the region -- the same "
                  "procedure used for training rollouts and imagined/DAgger rollouts"))
    out = IWS / "outputs/scaling_v2/scaling_viz.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"scaling payload: v2 {len(payload['v2'])} cells "
          f"({sum(len(v) for v in payload['v2'].values())} evals), "
          f"superseded {sum(len(v) for v in payload['v2_old'].values())}, "
          f"dagger {sum(len(v) for v in payload['dagger'].values())}, "
          f"trained {payload['v2_trained']}/60 + {payload['dagger_trained']}/48")


if __name__ == "__main__":
    main()

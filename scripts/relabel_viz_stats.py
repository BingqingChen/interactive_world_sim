"""Export the relabel-experiment quantification for the PairScope 'Relabel' tab.

Measures how much expert-sim relabeling changed the action labels of the
resynced r20+i400 mix: per-frame |a' - a| (raw label change, which includes
DDPM resampling noise), the drift-0 resampling floor (plan-1 chunks share the
exact conditioning between the two streams), the Stage-B CRN component (matched
noise -> pure hallucination-induced shift), the per-step motion scale, plus a
few per-episode traces and the sweep results table.

Writes outputs/relabel_mix/relabel_viz.json, consumed by export_pairscope_data.py.
"""
import json
from pathlib import Path
import sys

import numpy as np

DP_ROOT = "/home/jacobhb/projects/worth_doing/diffusion_policy"
sys.path.insert(0, DP_ROOT)
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402

R, N_REAL = 64, 20
IWS = Path(__file__).resolve().parent.parent


def main():
    pair = ReplayBuffer.copy_from_path(
        str(IWS / "datasets/rotate_t_r20_pair400_dp.zarr"), keys=["action"])
    rel = ReplayBuffer.copy_from_path(
        str(IWS / "datasets/rotate_t_r20_relabel400_dp.zarr"), keys=["action"])

    d_by = {dr: [] for dr in range(0, R, 8)}
    diffs, motions, traces = [], [], {}
    for ep in range(N_REAL, pair.n_episodes):
        a = pair.get_episode(ep)["action"]
        b = rel.get_episode(ep)["action"]
        d = np.linalg.norm(b - a, axis=1)
        diffs.append(d)
        motions.append(np.linalg.norm(np.diff(a, axis=0), axis=1))
        drift = (np.arange(len(d)) % R) // 8 * 8
        for dr in range(0, R, 8):
            d_by[dr].append(d[drift == dr])
        if ep in (25, 60, 138, 260, 333, 401):  # spread of lengths/shards
            traces[f"ep{ep}"] = np.round(d * 100, 2).tolist()

    alld = np.concatenate(diffs) * 100  # cm
    allm = np.concatenate(motions) * 100
    floor = np.concatenate(d_by[0]) * 100

    crn = {}
    for sh in ["D", "E"]:
        ad = np.load(IWS / f"datasets/pair{sh}_ad.npz")
        for t, mse in zip(ad["t"], ad["mse_exec"]):
            crn.setdefault(int(t) % R // 8 * 8, []).append(np.sqrt(mse) * 100)

    hist, edges = np.histogram(alld, bins=np.arange(0, 20.5, 0.5))
    payload = dict(
        n_frames=int(len(alld)),
        motion_cm=dict(mean=round(float(allm.mean()), 2),
                       median=round(float(np.median(allm)), 2)),
        raw_cm=dict(mean=round(float(alld.mean()), 2),
                    median=round(float(np.median(alld)), 2),
                    p90=round(float(np.percentile(alld, 90)), 2)),
        floor_cm=dict(mean=round(float(floor.mean()), 2),
                      median=round(float(np.median(floor)), 2)),
        by_drift={str(dr): dict(
            raw_mean=round(float(np.concatenate(d_by[dr]).mean() * 100), 2),
            raw_p90=round(float(np.percentile(np.concatenate(d_by[dr]) * 100, 90)), 2),
            crn_mean=(round(float(np.mean(crn[dr])), 2) if dr in crn else None),
        ) for dr in range(0, R, 8)},
        hist=dict(edges=edges.tolist(), counts=hist.tolist()),
        traces=traces,
        results=dict(
            i400_openloop=dict(seeds=[46, 38, 48], mean=44.0, sem=3.1),
            pair400=dict(seeds=[56, 20, 56], mean=44.0, sem=12.0),
            relab400=dict(seeds=[38, 42, 54], mean=44.7, sem=4.8),
            retr400=dict(seeds=[10, 0, 6], mean=5.3, sem=2.9),
            retrv400=dict(seeds=[2, 4, 0], mean=2.0, sem=1.2)),
        R=R)
    out = IWS / "outputs/relabel_mix/relabel_viz.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()

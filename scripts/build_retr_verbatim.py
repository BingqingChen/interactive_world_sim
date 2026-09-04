"""Build the verbatim-chunk variant of the STRAP-retrieval mix (sc_r20retrv400).

Reuses the DTW matches from build_retrieval_mix.py (retr400_matches.npz): at
every plan boundary t of an imagined episode, look up the matched real frame
m(t) and copy the expert's VERBATIM next 8 actions a_real[m(t):m(t)+8] (start
clamped so 8 actions fit inside the matched demo). Every DP 16-step target is
then two consecutive verbatim expert half-chunks at the expert's own clock --
mirroring build_relabeled_mix.py, so the arms differ only in label source.

Output: datasets/rotate_t_r20_retrv400_dp.zarr + retrv400_manifest.json

Usage: python scripts/build_retr_verbatim.py
"""
import json
import subprocess
import time
from pathlib import Path
import sys

import numpy as np

DP_ROOT = "/home/jacobhb/projects/worth_doing/diffusion_policy"
sys.path.insert(0, DP_ROOT)
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402

N_ACT, R = 8, 64
IWS = Path(__file__).resolve().parent.parent


def main():
    t0 = time.time()
    m = np.load(IWS / "datasets/retr400_matches.npz")
    match_ep, match_fr, lens = m["match_ep"], m["match_fr"], m["episode_lens"]
    off = np.r_[0, np.cumsum(lens[:-1])]

    real = ReplayBuffer.copy_from_path(str(IWS / "datasets/rotate_t_rand800_dp.zarr"))
    ee = np.array(real.episode_ends)
    starts = np.r_[0, ee[:-1]]
    real_act = real.data["action"][:]
    ep_len_real = ee - starts

    rb = ReplayBuffer.create_empty_numpy()
    for i in range(20):
        ep = real.get_episode(i)
        rb.add_episode({k: ep[k] for k in ("img", "state", "action")})

    diff_by, n_by = {}, {}
    n_img = 0
    for shard in ["D", "E"]:
        wm = ReplayBuffer.copy_from_path(
            str(IWS / f"datasets/rotate_t_imagined_pair{shard}_dp.zarr"),
            keys=["img", "state", "action"])
        for e in range(wm.n_episodes):
            if n_img >= len(lens):
                break
            ep = wm.get_episode(e)
            L = len(ep["img"])
            assert L == lens[n_img]
            a2 = np.zeros((L, 4), np.float32)
            for t in range(0, L - 1, N_ACT):
                oe = int(match_ep[off[n_img] + t])
                fr = int(match_fr[off[n_img] + t])
                fr = min(fr, int(ep_len_real[oe]) - N_ACT)  # fit 8 verbatim actions
                hi = min(t + N_ACT, L)
                a2[t:hi] = real_act[starts[oe] + fr: starts[oe] + fr + hi - t]
            if L % N_ACT == 1 and L > 1:
                a2[L - 1] = a2[L - 2]
            d = np.linalg.norm(a2 - ep["action"], axis=1)
            drift = (np.arange(L) % R) // N_ACT * N_ACT
            for dr in range(0, R, N_ACT):
                diff_by[dr] = diff_by.get(dr, 0.0) + d[drift == dr].sum()
                n_by[dr] = n_by.get(dr, 0) + int((drift == dr).sum())
            rb.add_episode({"img": ep["img"], "state": ep["state"], "action": a2})
            n_img += 1

    rb.save_to_path(str(IWS / "datasets/rotate_t_r20_retrv400_dp.zarr"),
                    if_exists="replace")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=IWS).stdout.strip()
    manifest = dict(
        git_sha=sha, n_imagined=n_img, n_episodes=rb.n_episodes,
        n_steps=int(rb.n_steps),
        mean_abs_diff_by_plan_drift={
            str(k): round(diff_by[k] / n_by[k], 5) for k in sorted(diff_by)},
        wall_s=int(time.time() - t0))
    (IWS / "datasets/retrv400_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Build the expert-sim-relabeled r20+i400 mix (and its control).

The imagined data's learning signal becomes KL(pi_exp(a|s) || pi_theta(a|w)):
observations stay the WM's hallucinated frames, but the action stream is
replaced by what the demonstrator policy would do at the paired TRUE simulator
state. Relabeling mirrors the demonstrator's own replan-every-8 structure: at
every plan boundary t = 0, 8, ..., end-8, sample pi_ref ONCE (native DDPM-100)
at obs (S_{t-1}, S_t) + the stored pseudo-proprio, take the executed 8 actions
-> a'_t..a'_{t+7}. The result is temporally coherent within chunks and
replanned every 8 -- the same structure as real DP rollouts. Episodes keep
img = W frames and state = stored pseudo-proprio ((actual state, expert
action) = DAgger convention); only `action` changes.

Outputs:
  datasets/rotate_t_r20_relabel400_dp.zarr  20 real (rand800 0:20) + 400
      relabeled imagined episodes (pooled shards, e.g. D+E)
  datasets/rotate_t_r20_pair400_dp.zarr     control: same episodes, ORIGINAL
      stored actions
  datasets/relabel400_manifest.json         per-drift |a' - a| profile, args

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/build_relabeled_mix.py \
      --dp_ckpt <collector ckpt> --shards D,E [--limit 400]
"""
import argparse
import json
import subprocess
import time
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_dp_rotate_t as E  # noqa: E402

from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402

N_ACT = 8


@torch.no_grad()
def relabel_episode_batch(policy, sim_imgs, states, lens, device, bs=96):
    """Relabel a list of episodes. sim_imgs/states: lists of (L,128,128,3)/(L,4).
    Returns list of (L,4) relabeled action streams."""
    conds = []  # (ep_idx, t, img_prev, img_curr, st_prev, st_curr)
    for e, (si, st, L) in enumerate(zip(sim_imgs, states, lens)):
        end = L - 1
        for t in range(0, end, N_ACT):
            p = max(t - 1, 0)
            conds.append((e, t, si[p], si[t], st[p], st[t]))
    out = [np.zeros((L, 4), np.float32) for L in lens]
    for i in range(0, len(conds), bs):
        chunk = conds[i:i + bs]
        img = torch.from_numpy(np.stack(
            [np.stack([c[2], c[3]]) for c in chunk])).to(device)
        img = img.permute(0, 1, 4, 2, 3).float() / 255.0
        pos = torch.from_numpy(np.stack(
            [np.stack([c[4], c[5]]) for c in chunk])).to(device).float()
        act = policy.predict_action({"image": img, "agent_pos": pos})["action"]
        act = act.cpu().numpy()  # (b, 8, 4) executed slice
        for (e, t, *_), a in zip(chunk, act):
            L = lens[e]
            hi = min(t + N_ACT, L)
            out[e][t:hi] = a[: hi - t]
    for e, L in enumerate(lens):  # final slot (pad convention: repeat last)
        if L % N_ACT == 1 and L > 1:
            out[e][L - 1] = out[e][L - 2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp_ckpt", required=True)
    ap.add_argument("--shards", default="D,E")
    ap.add_argument("--pair_prefix", default="datasets/")
    ap.add_argument("--real_zarr", default="datasets/rotate_t_rand800_dp.zarr")
    ap.add_argument("--n_real", type=int, default=20)
    ap.add_argument("--limit", type=int, default=400, help="imagined episodes")
    ap.add_argument("--ep_batch", type=int, default=25)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    pre = Path(args.pair_prefix)

    policy, n_obs, n_act = E.load_policy(args.dp_ckpt, device)
    assert n_obs == 2 and n_act == N_ACT

    rb_rel = ReplayBuffer.create_empty_numpy()
    rb_ctl = ReplayBuffer.create_empty_numpy()
    real = ReplayBuffer.copy_from_path(args.real_zarr)
    for i in range(args.n_real):
        ep = real.get_episode(i)
        ep = {k: ep[k] for k in ("img", "state", "action")}
        rb_rel.add_episode(ep)
        rb_ctl.add_episode(dict(ep))
    print(f"added {args.n_real} real episodes")

    # per-drift |a' - a| accounting (drift = frame index mod R within cycle)
    R = None
    diff_by_drift, n_by_drift = {}, {}
    n_img = 0
    t0 = time.time()
    for shard in args.shards.split(","):
        if n_img >= args.limit:
            break
        man = json.loads((pre / f"pair{shard}_meta.json").read_text())
        R = man["resync_period"]
        wm = ReplayBuffer.copy_from_path(
            str(pre / f"rotate_t_imagined_pair{shard}_dp.zarr"),
            keys=["img", "state", "action"])
        sim = ReplayBuffer.copy_from_path(
            str(pre / f"rotate_t_simreplay_pair{shard}_dp.zarr"), keys=["img"])
        eps = list(range(min(wm.n_episodes, args.limit - n_img)))
        for lo in range(0, len(eps), args.ep_batch):
            batch = eps[lo:lo + args.ep_batch]
            wms = [wm.get_episode(e) for e in batch]
            sims = [sim.get_episode(e)["img"] for e in batch]
            lens = [len(w["img"]) for w in wms]
            rel = relabel_episode_batch(
                policy, sims, [w["state"] for w in wms], lens, device)
            for w, a2 in zip(wms, rel):
                rb_rel.add_episode({"img": w["img"], "state": w["state"],
                                    "action": a2})
                rb_ctl.add_episode({"img": w["img"], "state": w["state"],
                                    "action": w["action"]})
                d = np.linalg.norm(a2 - w["action"], axis=1)
                for k in range(len(d)):
                    dr = (k % R) // N_ACT * N_ACT
                    diff_by_drift[dr] = diff_by_drift.get(dr, 0.0) + d[k]
                    n_by_drift[dr] = n_by_drift.get(dr, 0) + 1
            n_img += len(batch)
            print(f"  {shard}: {n_img}/{args.limit} imagined episodes "
                  f"({time.time() - t0:.0f}s)", flush=True)

    rb_rel.save_to_path(str(pre / "rotate_t_r20_relabel400_dp.zarr"),
                        if_exists="replace")
    rb_ctl.save_to_path(str(pre / "rotate_t_r20_pair400_dp.zarr"),
                        if_exists="replace")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=Path(__file__).parent).stdout.strip()
    manifest = dict(
        vars(args), git_sha=sha, R=R, n_imagined=n_img,
        n_episodes=rb_rel.n_episodes, n_steps=int(rb_rel.n_steps),
        mean_abs_diff_by_plan_drift={
            str(k): round(diff_by_drift[k] / n_by_drift[k], 5)
            for k in sorted(diff_by_drift)},
        wall_s=int(time.time() - t0))
    (pre / "relabel400_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

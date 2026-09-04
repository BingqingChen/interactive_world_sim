"""Score each imagined training window's action-distribution shift under the
demonstrator policy itself (Stage B of the H1/H2 study).

For every hallucinated-plan window produced by collect_imagined_rotate_t_paired.py
(obs = (W_{t-1}, W_t)), compare the demonstrator's action distribution under the
hallucinated obs q = pi_ref(a | W) against the counterfactual true-state
distribution p = pi_ref(a | S) at the same timestep, with agent_pos held to the
same pseudo-proprio in both branches (the frames are the only difference):

  mse   : common-random-numbers estimate of the conditioning-induced shift --
          K matched-noise DDIM(eta=0) sample pairs, mean ||a_p - a_q||^2 over the
          full 16x4 horizon (the training target). Raw action units (meters^2).
  kl    : forward KL(p || q) via the Kozachenko-Leonenko k-NN estimator on the
          executed-8 slice flattened to d=32 (d=64 is too biased for kNN).
          Ranks only; calibrate against `kl_floor`.
  kl_floor : KL(p || p') between two independent K-sample sets from the SAME
          s-branch, on a --floor_frac subset -- the estimator's zero point.

Window rule (must match the collector; R = resync_period from the meta json):
plan-start frames t = 8k with t % R != 0 (plan 1 of a cycle sees real obs),
(t + 8) % R != 0 (the cycle's last plan's 16-action target crosses the resync),
and t <= end - 16 (the target must not need actions past the episode).

All K raw samples from both branches are saved (pair<shard>_ad_samples.npz) so
any alternative divergence/slice/whitening can be recomputed with zero new
policy forwards.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/score_chunk_ad.py \
      --dp_ckpt <collector ckpt> --shard D [--k_samples 64] [--n_infer 16]
"""
import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_dp_rotate_t as E  # noqa: E402  (load_policy; puts DP_ROOT on sys.path)

from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


def enumerate_windows(ep_len, R):
    """Plan-start frames of the trainable hallucinated windows for one episode."""
    end = ep_len - 1  # last frame index; multiple of 8 by construction
    ts = []
    for t in range(8, end - 15, 8):
        if t % R == 0 or (t + 8) % R == 0:
            continue
        ts.append(t)
    return ts


@torch.no_grad()
def sample_actions(policy, imgs_u8, agent_pos, K, seed, device):
    """imgs_u8 (2,128,128,3), agent_pos (2,4) -> (K,16,4) action-horizon samples.
    Fixed seed => identical noise draws across calls with the same K (CRN)."""
    img = torch.from_numpy(imgs_u8).to(device).permute(0, 3, 1, 2).float() / 255.0
    obs = {
        "image": img.unsqueeze(0).expand(K, -1, -1, -1, -1),
        "agent_pos": torch.from_numpy(agent_pos).to(device)
                          .unsqueeze(0).expand(K, -1, -1).float(),
    }
    torch.manual_seed(seed)
    return policy.predict_action(obs)["action_pred"].cpu().numpy()  # (K,16,4)


def knn_kl(p, q, k=5):
    """Kozachenko-Leonenko KL(p||q) from samples p (n,d), q (m,d)."""
    n, d = p.shape
    m = q.shape[0]
    tp = torch.from_numpy(p)
    tq = torch.from_numpy(q)
    r = torch.cdist(tp, tp).topk(k + 1, largest=False).values[:, k]  # kth NN, self excluded
    s = torch.cdist(tp, tq).topk(k, largest=False).values[:, k - 1]
    r = torch.clamp(r, min=1e-12)
    s = torch.clamp(s, min=1e-12)
    return float(d * torch.log(s / r).mean() + np.log(m / (n - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp_ckpt", required=True, help="pi_ref = the demonstrator ckpt")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--pair_prefix", default="datasets/")
    ap.add_argument("--k_samples", type=int, default=64)
    ap.add_argument("--n_infer", type=int, default=16, help="DDIM steps (100 = DDPM-equal cost)")
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument("--floor_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    pre = Path(args.pair_prefix)
    manifest = json.loads((pre / f"pair{args.shard}_meta.json").read_text())
    R = manifest["resync_period"]

    policy, n_obs, n_act = E.load_policy(args.dp_ckpt, device)
    assert n_obs == 2 and n_act == 8
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    policy.noise_scheduler = DDIMScheduler.from_config(policy.noise_scheduler.config)
    policy.num_inference_steps = args.n_infer

    rb_wm = ReplayBuffer.copy_from_path(
        str(pre / f"rotate_t_imagined_pair{args.shard}_dp.zarr"),
        keys=["img", "state", "action"])
    rb_sim = ReplayBuffer.copy_from_path(
        str(pre / f"rotate_t_simreplay_pair{args.shard}_dp.zarr"), keys=["img"])
    assert rb_wm.n_episodes == rb_sim.n_episodes

    rng = np.random.default_rng(args.seed)
    rows, p_all, q_all = [], [], []
    t0 = time.time()
    n_win = 0
    for ep in range(rb_wm.n_episodes):
        wm = rb_wm.get_episode(ep)
        sim = rb_sim.get_episode(ep)
        L = len(wm["img"])
        for t in enumerate_windows(L, R):
            prev, curr = t - 1, t
            pseudo = wm["state"][[prev, curr]]  # same proprio in both branches
            seed = args.seed * 1_000_003 + n_win
            q = sample_actions(policy, wm["img"][[prev, curr]], pseudo,
                               args.k_samples, seed, device)
            p = sample_actions(policy, sim["img"][[prev, curr]], pseudo,
                               args.k_samples, seed, device)
            ex = slice(1, 1 + n_act)  # executed slice of the horizon
            mse = float(((p - q) ** 2).sum(axis=2).mean())        # per-step sq. err, full 16
            mse_exec = float(((p[:, ex] - q[:, ex]) ** 2).sum(axis=2).mean())
            kl = knn_kl(p[:, ex].reshape(args.k_samples, -1),
                        q[:, ex].reshape(args.k_samples, -1), args.knn_k)
            floor = np.nan
            if rng.random() < args.floor_frac:
                p2 = sample_actions(policy, sim["img"][[prev, curr]], pseudo,
                                    args.k_samples, seed + 500_000, device)
                floor = knn_kl(p[:, ex].reshape(args.k_samples, -1),
                               p2[:, ex].reshape(args.k_samples, -1), args.knn_k)
            rows.append((ep, t, mse, mse_exec, kl, floor))
            p_all.append(p.astype(np.float32))
            q_all.append(q.astype(np.float32))
            n_win += 1
        if ep % 10 == 0:
            print(f"  ep {ep}/{rb_wm.n_episodes}: {n_win} windows, "
                  f"{time.time() - t0:.0f}s", flush=True)

    rows = np.array(rows)
    out = pre / f"pair{args.shard}_ad.npz"
    np.savez_compressed(
        out, episode=rows[:, 0].astype(int), t=rows[:, 1].astype(int),
        mse=rows[:, 2], mse_exec=rows[:, 3], kl=rows[:, 4], kl_floor=rows[:, 5])
    np.savez_compressed(pre / f"pair{args.shard}_ad_samples.npz",
                        p=np.stack(p_all), q=np.stack(q_all),
                        episode=rows[:, 0].astype(int), t=rows[:, 1].astype(int))
    fl = rows[:, 5][np.isfinite(rows[:, 5])]
    summary = dict(shard=args.shard, n_windows=n_win, k_samples=args.k_samples,
                   n_infer=args.n_infer, knn_k=args.knn_k, R=R,
                   mse_median=float(np.median(rows[:, 2])),
                   kl_median=float(np.median(rows[:, 4])),
                   kl_floor_median=float(np.median(fl)) if len(fl) else None,
                   spearman_mse_kl=float(__import__("scipy.stats", fromlist=["x"])
                                         .spearmanr(rows[:, 2], rows[:, 4]).statistic),
                   wall_s=int(time.time() - t0))
    (pre / f"pair{args.shard}_ad.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

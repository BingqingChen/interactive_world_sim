"""Build the STRAP-retrieval-relabeled r20+i400 mix (baseline sc_r20retr400).

Imitation target for imagined data becomes pi_expert(a | s_closest), with
s_closest found by STRAP-style sub-trajectory retrieval over the FULL rand800
real-demo pool (github.com/WEIRDLabUW/STRAP; implementation details mirrored):

  - Encoder: DINOv2-base (HF facebook/dinov2-base), last_hidden_state
    avg-pooled over all tokens, UNnormalized; Euclidean distance.
  - Query segmentation: each imagined episode's pseudo-proprio stream is split
    by the STRAP stop heuristic (sum |d state| < threshold across the 4 EE-xy
    dims), short segments merged (min_subtraj_len, STRAP default 20).
  - Matching: subsequence-DTW (step sizes {1,2}; STRAP's *_dtw_21 functions,
    vendored verbatim) of each query segment against every real trajectory;
    the top-1 (lowest accumulated cost) match wins.
  - Relabeling: the DTW warping path aligns every query frame n to a retrieved
    real frame m(n); the label stream is a'[n] = a_real[match][m(n)] -- the
    retrieved expert sub-trajectory's actions, time-warped onto our episode.

Frames and state stay the imagined episode's own (img = W frames, state =
pseudo-proprio); only `action` changes, so training is the stock stride-1
scaling recipe (same convention as build_relabeled_mix.py).

Outputs:
  datasets/rotate_t_r20_retr400_dp.zarr   20 real (rand800 0:20) + 400 imagined
      episodes with retrieved-action streams
  datasets/retr400_matches.npz            per-frame matched (real_ep, real_frame),
      per-segment (episode, start, end, matched_ep, cost) -- for the viewer
  datasets/retr400_manifest.json          stats

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/build_retrieval_mix.py [--limit 400]
"""
import argparse
import json
import subprocess
import time
from pathlib import Path
import sys

import numba as nb
import numpy as np
import torch

DP_ROOT = "/home/jacobhb/projects/worth_doing/diffusion_policy"
sys.path.insert(0, DP_ROOT)
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402

N_ACT, R = 8, 64


# ---- STRAP subsequence-DTW (vendored from strap/utils/retrieval_utils.py) ----
@nb.jit(nopython=True)
def compute_accumulated_cost_matrix_subsequence_dtw_21(C):
    N, M = C.shape
    D = np.zeros((N + 1, M + 2))
    D[0:1, :] = np.inf
    D[:, 0:2] = np.inf
    D[1, 2:] = C[0, :]
    for n in range(1, N):
        for m in range(0, M):
            if n == 0 and m == 0:
                continue
            D[n + 1, m + 2] = C[n, m] + min(D[n, m + 1], D[n, m])
    return D[1:, 2:]


@nb.jit(nopython=True)
def compute_optimal_warping_path_subsequence_dtw_21(D, m=-1):
    N, M = D.shape
    n = N - 1
    if m < 0:
        m = D[N - 1, :].argmin()
    P = [(n, m)]
    while n > 0:
        if m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n - 1, m - 1], D[n - 1, m - 2])
            if val == D[n - 1, m - 1]:
                cell = (n - 1, m - 1)
            else:
                cell = (n - 1, m - 2)
        P.append(cell)
        n, m = cell
    P.reverse()
    return np.array(P)


# ---- STRAP query segmentation (adapted: 4-dim bimanual EE-xy state) ----
def segment_by_derivative(states, threshold, min_len):
    diff = np.abs(np.diff(states, axis=0)).sum(axis=1)
    stops = np.where(diff < threshold)[0]
    bounds, start = [], 0
    for s in stops:
        bounds.append((start, s + 1))
        start = s + 1
    if start < len(states):
        bounds.append((start, len(states)))
    # merge short segments forward (STRAP merge_short_segments semantics)
    merged, cur = [], bounds[0]
    for b in bounds[1:]:
        if cur[1] - cur[0] < min_len:
            cur = (cur[0], b[1])
        else:
            merged.append(cur)
            cur = b
    if cur[1] - cur[0] < min_len and merged:
        merged[-1] = (merged[-1][0], cur[1])
    else:
        merged.append(cur)
    return merged


IMAGENET = (torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


@torch.no_grad()
def dinov2_feats(model, processor, frames_u8, device, bs=256):
    """frames_u8 (N,128,128,3) -> (N,768) avg-pooled DINOv2 features (fp16).
    GPU replication of the HF processor pipeline (resize shortest-edge 256
    bicubic, center-crop 224, rescale, ImageNet normalize) -- the per-image
    PIL path is far too slow for 278k frames."""
    import torch.nn.functional as F
    mean, std = (t.to(device) for t in IMAGENET)
    out = []
    for i in range(0, len(frames_u8), bs):
        x = torch.from_numpy(np.ascontiguousarray(frames_u8[i:i + bs]))
        x = x.to(device).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=256, mode="bicubic", align_corners=False,
                          antialias=True).clamp(0, 1)
        x = ((x[:, :, 16:240, 16:240] - mean) / std)  # center crop 224
        f = model(pixel_values=x).last_hidden_state.mean(dim=1)  # avg pooling
        out.append(f.half())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="D,E")
    ap.add_argument("--pair_prefix", default="datasets/")
    ap.add_argument("--real_zarr", default="datasets/rotate_t_rand800_dp.zarr")
    ap.add_argument("--n_real_mix", type=int, default=20)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--seg_threshold", type=float, default=5e-3,
                    help="STRAP stop threshold on sum |d state| (4 EE-xy dims)")
    ap.add_argument("--min_subtraj_len", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--keys_cache", default="datasets/rand800_dinov2_keys.pt")
    args = ap.parse_args()

    device = torch.device(args.device)
    pre = Path(args.pair_prefix)
    t0 = time.time()

    from transformers import Dinov2Model, AutoImageProcessor
    model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)

    print("loading rand800 ...", flush=True)
    real = ReplayBuffer.copy_from_path(args.real_zarr)
    ee = np.array(real.episode_ends)
    starts = np.r_[0, ee[:-1]]
    if Path(args.keys_cache).exists():
        keys = torch.load(args.keys_cache, map_location=device)
        assert keys.shape[0] == real.n_steps
        print(f"  loaded cached DINOv2 keys {tuple(keys.shape)}")
    else:
        keys = dinov2_feats(model, processor, real.data["img"][:], device)
        torch.save(keys.cpu(), args.keys_cache)
        keys = keys.to(device)
        print(f"  encoded {real.n_steps} key frames in {time.time() - t0:.0f}s")
    real_act = real.data["action"][:]

    rb = ReplayBuffer.create_empty_numpy()
    for i in range(args.n_real_mix):
        ep = real.get_episode(i)
        rb.add_episode({k: ep[k] for k in ("img", "state", "action")})

    seg_rows = []          # (imag_ep_global, q_start, q_end, real_ep, r_start, r_end, cost_per_frame)
    match_ep_flat, match_fr_flat = [], []
    diff_by, n_by = {}, {}
    n_img = 0
    n_ep_offline = len(ee)
    for shard in args.shards.split(","):
        if n_img >= args.limit:
            break
        wm = ReplayBuffer.copy_from_path(
            str(pre / f"rotate_t_imagined_pair{shard}_dp.zarr"),
            keys=["img", "state", "action"])
        for e in range(min(wm.n_episodes, args.limit - n_img)):
            ep = wm.get_episode(e)
            L = len(ep["img"])
            q = dinov2_feats(model, processor, ep["img"], device)  # (L,768)
            # Euclidean distance matrix to ALL offline frames at once (GPU)
            dm_full = torch.cdist(q.float(), keys.float()).cpu().numpy()  # (L,N)
            a2 = np.zeros((L, 4), np.float32)
            m_ep = np.full(L, -1, np.int32)
            m_fr = np.full(L, -1, np.int32)
            for (qs, qe) in segment_by_derivative(
                    ep["state"], args.seg_threshold, args.min_subtraj_len):
                best = (np.inf, None)
                for oe in range(n_ep_offline):
                    C = dm_full[qs:qe, starts[oe]:ee[oe]]
                    if C.shape[0] > C.shape[1]:
                        continue
                    D = compute_accumulated_cost_matrix_subsequence_dtw_21(
                        np.ascontiguousarray(C, np.float64))
                    end_m = D[-1, :].argmin()
                    cost = D[-1, end_m]
                    if cost < best[0]:
                        best = (cost, (oe, D))
                oe, D = best[1]
                P = compute_optimal_warping_path_subsequence_dtw_21(D)
                for (n_q, m_o) in P:
                    fi = qs + n_q
                    gi = starts[oe] + max(m_o, 0)
                    a2[fi] = real_act[gi]
                    m_ep[fi] = oe
                    m_fr[fi] = max(m_o, 0)
                seg_rows.append((n_img, qs, qe, oe, int(max(P[0, 1], 0)),
                                 int(P[-1, 1]) + 1, float(best[0] / (qe - qs))))
            d = np.linalg.norm(a2 - ep["action"], axis=1)
            drift = (np.arange(L) % R) // N_ACT * N_ACT
            for dr in range(0, R, N_ACT):
                diff_by[dr] = diff_by.get(dr, 0.0) + d[drift == dr].sum()
                n_by[dr] = n_by.get(dr, 0) + int((drift == dr).sum())
            rb.add_episode({"img": ep["img"], "state": ep["state"], "action": a2})
            match_ep_flat.append(m_ep)
            match_fr_flat.append(m_fr)
            n_img += 1
            if n_img % 20 == 0:
                print(f"  {n_img}/{args.limit} imagined eps, {len(seg_rows)} segments "
                      f"({time.time() - t0:.0f}s)", flush=True)

    rb.save_to_path(str(pre / "rotate_t_r20_retr400_dp.zarr"), if_exists="replace")
    seg = np.array(seg_rows)
    np.savez_compressed(
        pre / "retr400_matches.npz",
        seg=seg, match_ep=np.concatenate(match_ep_flat),
        match_fr=np.concatenate(match_fr_flat),
        episode_lens=np.array([len(x) for x in match_ep_flat]))
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=Path(__file__).parent).stdout.strip()
    manifest = dict(
        vars(args), git_sha=sha, n_imagined=n_img, n_segments=len(seg_rows),
        n_episodes=rb.n_episodes, n_steps=int(rb.n_steps),
        seg_len=dict(mean=round(float((seg[:, 2] - seg[:, 1]).mean()), 1),
                     median=int(np.median(seg[:, 2] - seg[:, 1]))),
        cost_per_frame=dict(mean=round(float(seg[:, 6].mean()), 2),
                            p10=round(float(np.percentile(seg[:, 6], 10)), 2),
                            p90=round(float(np.percentile(seg[:, 6], 90)), 2)),
        mean_abs_diff_by_plan_drift={
            str(k): round(diff_by[k] / n_by[k], 5) for k in sorted(diff_by)},
        wall_s=int(time.time() - t0))
    (pre / "retr400_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Collect PAIRED imagined rotate-T demonstrations: DP-in-world-model rollouts with
the real simulator executing the identical actions in lockstep, and the WM fully
re-synced to the simulator every `--resync_period` frames.

Protocol (v3, "fully-resynced"), per resync cycle of R frames (R/8 plans):
  - Resync: the imagination is overwritten by reality. The WM latent context is
    rebuilt from the last <=10 REAL sim frames, and the policy's obs buffer is set
    to the last two REAL sim frames (+ true EE proprio). Plan 1 of each cycle
    therefore acts on ground truth (belief-reality gap 0).
  - Plans 2..R/8: the policy conditions on the WM's own decoded frames (+ pseudo
    proprio = previous commanded target), i.e. on a short-horizon *prediction of
    the current real state* (gap <= 8k steps at plan k+1). The DP predicts a
    16-action horizon and executes 8, exactly as at deployment.
  - After the cycle's plans are generated, the simulator executes the identical
    commanded EE-xy targets through the standard PID/IK controller, continuing the
    episode's physics from its persistent state (per-episode snapshots carry the
    physics across the batched cycle loop).
  - Termination is checked on the sim's ground-truth T angle at every PLAN
    boundary (8 frames): the episode ends at the first plan boundary at/after the
    -80 deg CW crossing. Episodes that never cross within --ep_len (or end shorter
    than --min_len) are discarded, mirroring the original collector.

Every action is thus predicted for the state it is executed on (up to the WM's
k-step prediction error), and every WM frame is paired with a same-index sim frame.

Outputs (per shard):
  <out_prefix>rotate_t_imagined_pair<shard>_dp.zarr   img = [S_0, W_1..W_end]
  <out_prefix>rotate_t_simreplay_pair<shard>_dp.zarr  img = [S_0, S_1..S_end]
  both with: state = pseudo-proprio (previous commanded target, the training
  convention), state_sim = true EE-xy from the simulator, action = commanded
  targets (action[k] executed at frame k). Training reads only img/state/action.
  <out_prefix>pair<shard>_meta.npz  per-episode: init physics snapshot,
  world_t_bases, env reset seed, episode length, terminal frame, worst
  snapshot-restore PSNR; per-frame (flat + episode_ends offsets): PSNR(W,S),
  sim ground-truth T angle; per-cycle: decoder round-trip PSNR floor.
  <out_prefix>pair<shard>_meta.json  run manifest (args, counts, git SHA).

SSIM/LPIPS and any alternative per-window metrics are computed downstream from
the two frame zarrs (all raw frames are preserved), not here.

Usage:
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python scripts/collect_imagined_rotate_t_paired.py \
      --dp_ckpt <collector ckpt> --shard D --env_seed_base 600000 \
      --n_episodes 200 --ep_len 400 --batch 25 --resync_period 64 --random_init
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "data_collection"))

import eval_dp_rotate_t as E  # noqa: E402  (load_policy, get_obs, controller gains)
import eval_metric_correlation as M  # noqa: E402  (load_viz_cfg, load_model)
import eval_wm_quality as W  # noqa: E402  (psnr_series, sample_initial_state)
import collect_imagined_rotate_t as CI  # noqa: E402  (TERMINAL_DEG)
import collect_rotate_t as C  # noqa: E402
import gym_aloha.env as gae  # noqa: E402
from gym_aloha.env import AlohaEnv  # noqa: E402
from yixuan_utilities.kinematics_helper import KinHelper  # noqa: E402

from sim_aloha_dataset_collection_scripted import (  # noqa: E402
    trajectory_to_joint_actions,
)
from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm  # noqa: E402

from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402

TERMINAL_RAD = np.radians(CI.TERMINAL_DEG)  # -80 deg (CW negative)
N_ACT = 8  # asserted against the checkpoint below
CTX = 10  # WM latent context length (matches collect_imagined_rotate_t)


@torch.no_grad()
def encode_frames(wm, frames_u8, device):
    """frames_u8 (B,T,128,128,3) uint8 -> latents (B,T,C,H,W) in wm.dtype."""
    b, t = frames_u8.shape[:2]
    x = torch.from_numpy(frames_u8.reshape(b * t, 128, 128, 3)).to(device)
    x = x.permute(0, 3, 1, 2).float() / 255.0
    z = wm.encoder_forward(wm.normalizer["top_pov"].normalize(x)).to(wm.dtype)
    return z.reshape(b, t, *z.shape[1:])


@torch.no_grad()
def decode_latents(wm, z, bsz, n):
    """z (B,n,C,H,W) -> decoded frames, uint8 (B,n,128,128,3) and float (B,n,3,128,128)."""
    dec = render_img_cm(
        wm, z.reshape(bsz * n, *z.shape[2:]), resolution=128,
        normalizer=wm.normalizer, num_views=1, batch_size=16,
    ).float().clamp(0, 1).reshape(bsz, n, 3, 128, 128)
    u8 = (dec * 255).round().byte().permute(0, 1, 3, 4, 2).cpu().numpy()
    return u8, dec


def restore_physics(env, snap):
    env.reset(seed=0)
    env._env.physics.set_state(snap)
    env._env.physics.forward()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp_ckpt", required=True)
    ap.add_argument("--wm_ckpt", default="ckpts/push_t/epoch=3-step=90000.ckpt")
    ap.add_argument("--wm_config", default="pusht_mujoco")
    ap.add_argument("--shard", required=True, help="shard tag, e.g. D")
    ap.add_argument("--n_episodes", type=int, default=200)
    ap.add_argument("--ep_len", type=int, default=400)
    ap.add_argument("--resync_period", type=int, default=64,
                    help="R: frames between full WM<-sim resyncs. Multiple of 8, >=16.")
    ap.add_argument("--min_len", type=int, default=40)
    ap.add_argument("--fail_fast_frame", type=int, default=224,
                    help="Reject an episode early if by this frame the sim T has not "
                         "rotated past --fail_fast_deg CW (checked at cycle boundaries; "
                         "mirrors the real collector's insufficient-progress bailout).")
    ap.add_argument("--fail_fast_deg", type=float, default=-20.0)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--out_prefix", default="datasets/")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--random_init", action="store_true")
    ap.add_argument("--env_seed_base", type=int, default=600_000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    R = args.resync_period
    assert R % N_ACT == 0 and R >= 2 * N_ACT, "resync_period must be a multiple of 8, >=16"
    assert args.ep_len % N_ACT == 0

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading world model from {args.wm_ckpt} ...")
    cfg = M.load_viz_cfg(args.wm_config)
    wm = M.load_model(args.wm_ckpt, cfg.algorithm, str(device))
    wm.dec_infer_steps = 1  # match the original i400 data-generating process

    print(f"Loading DP policy from {args.dp_ckpt} ...")
    policy, n_obs, n_act = E.load_policy(args.dp_ckpt, device)
    assert n_obs == 2 and n_act == N_ACT, (n_obs, n_act)

    env = AlohaEnv("pusht")
    kin = KinHelper(robot_name="trossen_vx300s")
    rng_xy = (-0.08, 0.08) if args.random_init else (0.0, 0.0)
    gae.sample_pusht_pose = C.make_upright_pose_sampler(rng_xy, rng_xy)
    print(f"init: {'random XY +-0.08 + settle' if args.random_init else 'fixed'} | "
          f"R={R} ({R // N_ACT} plans/cycle) | ep_len={args.ep_len}")

    rb_wm = ReplayBuffer.create_empty_numpy()
    rb_sim = ReplayBuffer.create_empty_numpy()
    meta = {k: [] for k in ("init_snap", "world_t_bases", "env_seed", "ep_len",
                            "terminal_frame", "worst_restore_psnr")}
    psnr_flat, angle_flat, floor_flat = [], [], []
    ep_ends, floor_counts = [], []

    n_done = n_attempted = batch_i = 0
    env_seed = args.env_seed_base + args.seed
    t_start = time.time()

    while n_done < args.n_episodes:
        bsz = args.batch
        t0 = time.time()

        # ---- per-episode init: reset + snapshot (reuses the wm-quality protocol) ----
        init = []
        for _ in range(bsz):
            image, agent_pos, frame_u8, wtb, snap, env_seed = W.sample_initial_state(
                env, kin, env_seed, args.random_init)
            init.append((image, agent_pos, frame_u8, wtb, snap, env_seed - 1))
        frame0_u8 = np.stack([i[2] for i in init])          # (B,128,128,3)
        home_xy = np.stack([i[1] for i in init]).astype(np.float32)  # (B,4)
        wtbs = [i[3] for i in init]
        snaps = [i[4].copy() for i in init]                  # persistent physics state
        init_snaps = [i[4].copy() for i in init]
        seeds = [i[5] for i in init]

        # ---- episode-wide buffers ----
        W_imgs = np.zeros((bsz, args.ep_len + 1, 128, 128, 3), np.uint8)
        S_imgs = np.zeros_like(W_imgs)
        W_imgs[:, 0] = frame0_u8
        S_imgs[:, 0] = frame0_u8
        state_sim = np.zeros((bsz, args.ep_len + 1, 4), np.float32)
        state_sim[:, 0] = home_xy
        gt_angle = np.full((bsz, args.ep_len + 1), np.nan)
        for b in range(bsz):
            restore_physics(env, snaps[b])
            gt_angle[b, 0] = C.t_angle(env)
        psnr = np.full((bsz, args.ep_len + 1), np.nan)
        floors = [[] for _ in range(bsz)]
        # action slot k = target executed at frame k (+N_ACT slack for the pad slot)
        actions = torch.zeros(bsz, args.ep_len + N_ACT, 4)
        curr_vel = np.zeros((bsz, 6))
        active = np.ones(bsz, bool)
        term_frame = np.full(bsz, -1, int)
        worst_restore = np.full(bsz, np.inf)

        policy.reset()
        t = 0  # index of the latest frame that exists (0 = reset frame)
        while t < args.ep_len and active.any():
            cyc_start = t
            n_plans = min(R, args.ep_len - t) // N_ACT

            # ---- resync: latent context and obs buffer from REAL sim frames ----
            lo = max(0, t - (CTX - 1))
            z_hist = encode_frames(wm, S_imgs[:, lo:t + 1], device)
            # decoder round-trip floor on the anchor frame
            u8_rt, _ = decode_latents(wm, z_hist[:, -1:], bsz, 1)
            for b in range(bsz):
                if active[b]:
                    floors[b].append(W.psnr_series(u8_rt[b], S_imgs[b, t:t + 1])[0])
            if t == 0:
                prev_u8, curr_u8 = S_imgs[:, 0], S_imgs[:, 0]
                prev_st, curr_st = home_xy, home_xy
            else:
                prev_u8, curr_u8 = S_imgs[:, t - 1], S_imgs[:, t]
                prev_st, curr_st = state_sim[:, t - 1], state_sim[:, t]  # true EE

            # ---- generate the cycle's plans (GPU, batched over episodes) ----
            for p in range(n_plans):
                obs = {
                    "image": torch.stack([
                        torch.from_numpy(prev_u8).to(device).permute(0, 3, 1, 2).float() / 255.0,
                        torch.from_numpy(curr_u8).to(device).permute(0, 3, 1, 2).float() / 255.0,
                    ], dim=1),
                    "agent_pos": torch.stack([
                        torch.from_numpy(np.ascontiguousarray(prev_st)).to(device),
                        torch.from_numpy(np.ascontiguousarray(curr_st)).to(device),
                    ], dim=1),
                }
                with torch.no_grad():
                    plan = policy.predict_action(obs)["action"]  # (B,8,4)
                actions[:, t:t + N_ACT] = plan.cpu().float()
                actions[:, t + N_ACT] = plan[:, -1].cpu().float()  # frame-align pad

                act_lo = max(0, t - (z_hist.shape[1] - 1))
                assert z_hist.shape[1] == t - act_lo + 1
                act_win = actions[:, act_lo:t + N_ACT + 1].to(device)
                act_win = wm.normalizer["action"].normalize(act_win).to(wm.dtype)
                with torch.no_grad():
                    z_new = wm.dynamics_forward(z_hist, act_win)  # (B,8,C,H,W)
                z_hist = torch.cat([z_hist, z_new], dim=1)[:, -CTX:]
                u8, _ = decode_latents(wm, z_new, bsz, N_ACT)
                W_imgs[:, t + 1:t + 1 + N_ACT] = u8
                t += N_ACT
                prev_u8, curr_u8 = W_imgs[:, t - 1], W_imgs[:, t]
                prev_st = actions[:, t - 2].numpy()  # pseudo-proprio for W-frame plans
                curr_st = actions[:, t - 1].numpy()

            # ---- simulator executes the identical actions (CPU, per episode) ----
            for b in range(bsz):
                if not active[b]:
                    continue
                restore_physics(env, snaps[b])
                _, _, fr = E.get_obs(env, kin, wtbs[b])
                rp = W.psnr_series(fr[None], S_imgs[b, cyc_start:cyc_start + 1])[0]
                worst_restore[b] = min(worst_restore[b], rp)
                for k in range(cyc_start, t):  # action[k] produces frame k+1
                    o = env._env.task.get_observation(env._env.physics)
                    joint, _ = trajectory_to_joint_actions(
                        actions[b, k].numpy().astype(np.float64), wtbs[b], kin,
                        o["qpos"][:14], curr_vel[b], E.DT, E.K_P, E.K_V,
                        E.ACC_LIM, E.VEL_LIM)
                    env.step(joint)
                    _, ee, fr = E.get_obs(env, kin, wtbs[b])
                    S_imgs[b, k + 1] = fr
                    state_sim[b, k + 1] = ee
                    gt_angle[b, k + 1] = C.t_angle(env)
                snaps[b] = env._env.physics.get_state().copy()
                psnr[b, cyc_start + 1:t + 1] = W.psnr_series(
                    W_imgs[b, cyc_start + 1:t + 1], S_imgs[b, cyc_start + 1:t + 1])
                # terminal check at plan boundaries within this cycle
                crossed = np.where(gt_angle[b, cyc_start + 1:t + 1] <= TERMINAL_RAD)[0]
                if len(crossed):
                    f = cyc_start + 1 + crossed[0]
                    term_frame[b] = f
                    active[b] = False
                elif (t >= args.fail_fast_frame
                        and gt_angle[b, t] > np.radians(args.fail_fast_deg)):
                    active[b] = False  # insufficient progress -> early reject

        # ---- accept/reject and store ----
        n_acc = 0
        for b in range(bsz):
            n_attempted += 1
            if n_done >= args.n_episodes:
                break
            if term_frame[b] < 0:
                continue  # never crossed -80 deg -> discard
            end = int(np.ceil(term_frame[b] / N_ACT)) * N_ACT  # plan boundary >= crossing
            if end + 1 < args.min_len:
                continue
            pseudo = np.zeros((end + 1, 4), np.float32)
            pseudo[0] = home_xy[b]
            pseudo[1:] = actions[b, :end].numpy()  # EE estimate = previous target
            ep_wm = {"img": W_imgs[b, :end + 1], "state": pseudo,
                     "state_sim": state_sim[b, :end + 1],
                     "action": actions[b, :end + 1].numpy()}
            ep_sim = dict(ep_wm, img=S_imgs[b, :end + 1])
            rb_wm.add_episode(ep_wm)
            rb_sim.add_episode(ep_sim)
            meta["init_snap"].append(init_snaps[b])
            meta["world_t_bases"].append(wtbs[b])
            meta["env_seed"].append(seeds[b])
            meta["ep_len"].append(end + 1)
            meta["terminal_frame"].append(int(term_frame[b]))
            meta["worst_restore_psnr"].append(float(worst_restore[b]))
            psnr_flat.append(psnr[b, :end + 1])
            angle_flat.append(gt_angle[b, :end + 1])
            floor_flat.append(np.array(floors[b]))
            floor_counts.append(len(floors[b]))
            ep_ends.append((ep_ends[-1] if ep_ends else 0) + end + 1)
            n_done += 1
            n_acc += 1
        print(f"  batch {batch_i}: accepted {n_acc}/{bsz} in {time.time() - t0:.0f}s "
              f"({n_done}/{args.n_episodes}, accept {n_done / max(n_attempted, 1):.2f})",
              flush=True)
        batch_i += 1

    out = Path(args.out_prefix)
    zw = str(out / f"rotate_t_imagined_pair{args.shard}_dp.zarr")
    zs = str(out / f"rotate_t_simreplay_pair{args.shard}_dp.zarr")
    rb_wm.save_to_path(zw, if_exists="replace")
    rb_sim.save_to_path(zs, if_exists="replace")
    np.savez_compressed(
        out / f"pair{args.shard}_meta.npz",
        init_snap=np.stack(meta["init_snap"]),
        world_t_bases=np.stack(meta["world_t_bases"]),
        env_seed=np.array(meta["env_seed"]),
        ep_len=np.array(meta["ep_len"]),
        terminal_frame=np.array(meta["terminal_frame"]),
        worst_restore_psnr=np.array(meta["worst_restore_psnr"]),
        episode_ends=np.array(ep_ends),
        psnr=np.concatenate(psnr_flat),
        gt_angle=np.concatenate(angle_flat),
        decoder_floor=np.concatenate(floor_flat),
        decoder_floor_counts=np.array(floor_counts),
    )
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=Path(__file__).parent).stdout.strip()
    lens = np.array(meta["ep_len"])
    manifest = dict(vars(args), git_sha=sha, n_accepted=n_done, n_attempted=n_attempted,
                    accept_rate=round(n_done / n_attempted, 3),
                    ep_len_min=int(lens.min()), ep_len_median=int(np.median(lens)),
                    ep_len_max=int(lens.max()),
                    psnr_mean=float(np.nanmean(np.concatenate(psnr_flat))),
                    wall_s=int(time.time() - t_start))
    (out / f"pair{args.shard}_meta.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {n_done} paired episodes -> {zw} + {zs}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

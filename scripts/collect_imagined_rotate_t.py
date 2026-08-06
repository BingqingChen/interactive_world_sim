"""Collect "imagined" rotate-T demonstrations by rolling a trained Diffusion Policy
out INSIDE the pretrained IWS latent world model (no physics simulator in the loop).

Closed-loop imagination, batched over episodes:
  1. A single real env reset (fixed T position, fixed home arms, --no_settle style)
     provides the initial 128x128 top_pov frame + the home bimanual EE-xy.
  2. Repeat until ep_len frames exist:
       - DP predicts 8 target EE-xy actions from the last 2 imagined frames +
         pseudo-proprioception (previous commanded target = current EE estimate,
         valid because the PID controller tracks targets closely).
       - The WM dynamics generates the next 8 latents conditioned on the
         frame-aligned action sequence (the model pairs action[t] with frame t,
         so the not-yet-planned trailing slot is padded with the last action and
         overwritten when the next plan arrives).
       - The CM decoder renders the new latents to [0,1] images -> next DP obs.
  3. Episodes are stored directly in a Diffusion Policy zarr replay buffer
     (img uint8, state = pseudo EE-xy, action = DP targets), the exact format
     convert_rotate_t_to_dp_zarr.py produces for real demos.

Termination: a deterministic image-based T-angle estimator (red-mask template
rotation matching against the known upright frame-0 mask; validated at 0.8 deg
mean error vs ground truth on real demos) detects when the imagined T first
reaches ~80 deg clockwise. Episodes are truncated there, mirroring how the real
scripted collector ends demos at success; imagined episodes that never reach the
terminal angle within --ep_len frames are DISCARDED (like failed real trials).
Diversity across episodes comes from DP diffusion-sampling noise and WM dynamics
noise; all episodes share the same initial state, matching the fixed-init task.

Usage:
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 python scripts/collect_imagined_rotate_t.py \
      --dp_ckpt <path/to/best.ckpt> --n_episodes 200 --ep_len 400 --batch 25 \
      --out_zarr datasets/rotate_t_imagined_dp.zarr
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "data_collection"))

import eval_dp_rotate_t as E  # noqa: E402  (load_policy, get_obs, DP_ROOT on sys.path)
import eval_metric_correlation as M  # noqa: E402  (load_viz_cfg, load_model)
import collect_rotate_t as C  # noqa: E402
import gym_aloha.env as gae  # noqa: E402
from gym_aloha.env import AlohaEnv  # noqa: E402
from yixuan_utilities.kinematics_helper import KinHelper  # noqa: E402

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm  # noqa: E402
from interactive_world_sim.utils.pose_utils import PoseType, pose_convert  # noqa: E402

from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


# ----------------------------------------------------------------------------- #
# Image-based T-angle estimator (terminal detector for imagined episodes)
# ----------------------------------------------------------------------------- #
TERMINAL_DEG = -80.0  # first crossing of 80 deg CW ends an imagined episode
THETAS = np.radians(np.arange(-130, 41, 1.5))


def red_mask(img):
    """Segment the red T. img: (H,W,3) uint8 -> bool mask."""
    r = img[..., 0].astype(np.int16)
    g = img[..., 1].astype(np.int16)
    b = img[..., 2].astype(np.int16)
    return (r > 140) & (g < 110) & (b < 110) & (r - np.maximum(g, b) > 40)


def make_templates(frame0):
    """Rotate frame-0's (upright) T mask over the THETAS grid about its centroid."""
    import cv2
    m0 = red_mask(frame0).astype(np.uint8)
    ys, xs = np.nonzero(m0)
    cy, cx = ys.mean(), xs.mean()
    tpl = np.stack([
        cv2.warpAffine(m0, cv2.getRotationMatrix2D((cx, cy), np.degrees(th), 1.0),
                       (m0.shape[1], m0.shape[0])) > 0
        for th in THETAS
    ])  # (n_theta, H, W) bool
    return tpl, (cy, cx)


def est_angle(img, templates, tc):
    """Estimated T z-angle (rad) via best-IoU rotated template, or None if no T."""
    import cv2
    m = red_mask(img)
    if m.sum() < 30:
        return None
    ys, xs = np.nonzero(m)
    m_al = cv2.warpAffine(
        m.astype(np.uint8),
        np.float32([[1, 0, tc[1] - xs.mean()], [0, 1, tc[0] - ys.mean()]]),
        (m.shape[1], m.shape[0]),
    ) > 0
    inter = (templates & m_al).sum(axis=(1, 2))
    union = (templates | m_al).sum(axis=(1, 2))
    return float(THETAS[np.argmax(inter / np.maximum(union, 1))])


def find_terminal_frame(ep_imgs, templates, tc, stride=4):
    """First frame index where the T crosses TERMINAL_DEG (with a persistence
    check against single-frame hallucination flicker), or None."""
    term = np.radians(TERMINAL_DEG)
    n = len(ep_imgs)
    for k in range(0, n, stride):
        a = est_angle(ep_imgs[k], templates, tc)
        if a is not None and a <= term:
            # refine backward to the exact first crossing
            first = k
            for j in range(max(0, k - stride + 1), k):
                aj = est_angle(ep_imgs[j], templates, tc)
                if aj is not None and aj <= term:
                    first = j
                    break
            # persistence: still past ~75 deg a few frames later (or at the end)
            chk = min(first + 6, n - 1)
            ac = est_angle(ep_imgs[chk], templates, tc)
            if ac is not None and ac <= np.radians(-75.0):
                return first
    return None


def get_initial_states(env, kin, bsz, seed0, random_init, trial=0):
    """Per-episode real env resets -> batched initial frames + start EE-xy.

    random_init=False: T fixed at (0,0) upright, arms at home (no settle).
    random_init=True:  T at random XY (+-0.08) upright, random arm settle, with the
    same uprightness retry guard as the real varied-init collection/eval protocol.
    Returns (imgs (B,3,128,128) f32 [0,1], homes (B,4) f32, frames (B,128,128,3) u8,
    next trial counter)."""
    imgs, homes, frames = [], [], []
    for _ in range(bsz):
        while True:
            np.random.seed(seed0 + trial)
            env.reset(seed=seed0 + trial)
            trial += 1
            C.stabilize_t(env)  # settled start frame -- matches the WM's training frames
            o = env._env.task.get_observation(env._env.physics)
            lb = pose_convert(o["left_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            rb = pose_convert(o["right_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            world_t_bases = np.stack([lb, rb])
            if random_init:
                C.settle_arms(env, world_t_bases, kin, np.zeros(6),
                              E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM)
                if abs(C.t_angle(env)) > C.UPRIGHT_TOL:
                    continue  # settle bumped the T off upright -> fresh reset
            image, agent_pos, frame_uint8 = E.get_obs(env, kin, world_t_bases)
            imgs.append(image)
            homes.append(agent_pos)
            frames.append(frame_uint8)
            break
    return (np.stack(imgs), np.stack(homes).astype(np.float32),
            np.stack(frames), trial)


@torch.no_grad()
def imagine_batch(wm, policy, img0, home_xy, frame0_u8, bsz, ep_len, n_act, device):
    """Roll `bsz` episodes of `ep_len` frames inside the world model, each from its
    OWN initial state (img0 (B,3,128,128), home_xy (B,4), frame0_u8 (B,128,128,3)).
    Returns (imgs uint8 (B,ep_len,128,128,3), states (B,ep_len,4), actions (B,ep_len,4))."""
    assert ep_len % n_act == 0, "ep_len must be a multiple of n_action_steps"
    n_plans = ep_len // n_act

    # Frame store (uint8) and float [0,1] torch copies of the last 2 frames for DP.
    imgs = np.empty((bsz, ep_len, 128, 128, 3), dtype=np.uint8)
    imgs[:, 0] = frame0_u8
    states = np.empty((bsz, ep_len, 4), dtype=np.float32)
    states[:, 0] = home_xy
    # action slot k = target executed at frame k; +n_act slack for the trailing pad.
    actions = torch.zeros(bsz, ep_len + n_act, 4, dtype=torch.float32)

    # Encode each episode's own initial frame.
    f0 = torch.from_numpy(img0).to(device)  # (B,3,128,128) [0,1]
    f0n = wm.normalizer["top_pov"].normalize(f0)
    z0 = wm.encoder_forward(f0n).to(wm.dtype)  # (B,C,H,W)
    z_hist = z0.unsqueeze(1)  # (B,1,C,H,W)

    prev_img = f0.clone()  # (B,3,128,128)
    curr_img = f0.clone()
    prev_state = torch.from_numpy(home_xy).to(device)  # (B,4)
    curr_state = prev_state.clone()

    policy.reset()
    t = 0  # index of the current (latest) frame
    for plan_i in range(n_plans):
        # --- DP plan from the last 2 imagined frames + pseudo-proprioception ---
        obs_dict = {
            "image": torch.stack([prev_img, curr_img], dim=1),      # (B,2,3,128,128)
            "agent_pos": torch.stack([prev_state, curr_state], dim=1),  # (B,2,4)
        }
        plan = policy.predict_action(obs_dict)["action"]  # (B,n_act,4) raw EE-xy
        actions[:, t : t + n_act] = plan.cpu().float()
        actions[:, t + n_act] = plan[:, -1].cpu().float()  # trailing pad (overwritten next plan)

        # --- WM: generate the next n_act latents ---
        # z_hist always holds frames lo..t (last <=10), so actions must cover
        # slots lo..t+n_act to stay frame-aligned (dynamics pairs action[k] with
        # frame k; the slot at t+n_act is the pad written above).
        lo = max(0, t - 9)
        assert z_hist.shape[1] == t - lo + 1, (z_hist.shape, t, lo)
        act_win = actions[:, lo : t + n_act + 1].to(device)
        act_win = wm.normalizer["action"].normalize(act_win).to(wm.dtype)
        z_new = wm.dynamics_forward(z_hist, act_win)  # (B,n_act,C,H,W)
        z_hist = torch.cat([z_hist, z_new], dim=1)[:, -10:]

        # --- decode to images ---
        Bn = bsz * n_act
        dec = render_img_cm(
            wm, z_new.reshape(Bn, *z_new.shape[2:]), resolution=128,
            normalizer=wm.normalizer, num_views=1, batch_size=16,
        ).float().clamp(0, 1).reshape(bsz, n_act, 3, 128, 128)

        # --- store frames/states; advance obs history ---
        for j in range(n_act):
            k = t + 1 + j  # global index of this new frame
            if k >= ep_len:
                break
            imgs[:, k] = (
                (dec[:, j] * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            )
            states[:, k] = actions[:, k - 1].numpy()  # EE estimate = previous target
        prev_img = dec[:, -2] if n_act >= 2 else curr_img
        curr_img = dec[:, -1]
        t += n_act
        prev_state = actions[:, t - 2].to(device)
        curr_state = actions[:, t - 1].to(device)

    return imgs, states, actions[:, :ep_len].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp_ckpt", required=True)
    ap.add_argument("--wm_ckpt", default="ckpts/push_t/epoch=3-step=90000.ckpt")
    ap.add_argument("--wm_config", default="pusht_mujoco")
    ap.add_argument("--n_episodes", type=int, default=200)
    ap.add_argument("--ep_len", type=int, default=400,
                    help="Max imagined frames per episode (cap for the terminal detector).")
    ap.add_argument("--terminal_mode", choices=["angle", "fixed"], default="angle",
                    help="'angle': truncate at the image-estimated 80deg-CW crossing and "
                         "discard episodes that never cross; 'fixed': keep all, full length.")
    ap.add_argument("--min_len", type=int, default=40,
                    help="Discard imagined episodes shorter than this after truncation.")
    ap.add_argument("--save_rejects_dir", default=None,
                    help="If set, save every DISCARDED imagined episode as an mp4 here "
                         "(full ep_len rollout, final estimated T angle in the filename).")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--out_zarr", default="datasets/rotate_t_imagined_dp.zarr")
    ap.add_argument("--video_dir", default="outputs/imagined_videos")
    ap.add_argument("--n_videos", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--random_init", action="store_true",
                    help="Per-episode random T XY (+-0.08, upright) + random arm settle, "
                         "matching the varied-init real protocol. Default: fixed init.")
    ap.add_argument("--env_seed_base", type=int, default=500_000,
                    help="Base seed for the per-episode real env resets (kept disjoint "
                         "from the tight-eval seed 7000).")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading world model from {args.wm_ckpt} ...")
    cfg = M.load_viz_cfg(args.wm_config)
    wm = M.load_model(args.wm_ckpt, cfg.algorithm, str(device))
    wm.dec_infer_steps = 1

    print(f"Loading DP policy from {args.dp_ckpt} ...")
    policy, n_obs, n_act = E.load_policy(args.dp_ckpt, device)
    assert n_obs == 2, f"collector assumes n_obs_steps=2, got {n_obs}"
    print(f"  n_obs_steps={n_obs} n_action_steps={n_act}")

    # Real env: supplies each episode's initial frame/state (and only that --
    # every transition afterwards is imagined).
    env = AlohaEnv("pusht")
    kin = KinHelper(robot_name="trossen_vx300s")
    rng_xy = (-0.08, 0.08) if args.random_init else (0.0, 0.0)
    gae.sample_pusht_pose = C.make_upright_pose_sampler(rng_xy, rng_xy)
    print(f"init mode: {'RANDOM XY +-0.08 + arm settle' if args.random_init else 'fixed (0,0), no settle'}")

    rb = ReplayBuffer.create_empty_numpy()
    if args.video_dir:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_attempted = 0
    batch_i = 0
    env_trial = 0
    ep_lens, n_videos_written = [], 0
    while n_done < args.n_episodes:
        bsz = args.batch
        t0 = time.time()
        img0, home_xy, frame0_u8, env_trial = get_initial_states(
            env, kin, bsz, args.env_seed_base + args.seed, args.random_init, env_trial
        )
        imgs, states, actions = imagine_batch(
            wm, policy, img0, home_xy, frame0_u8, bsz, args.ep_len, n_act, device
        )
        n_acc = 0
        for b in range(bsz):
            n_attempted += 1
            if n_done >= args.n_episodes:
                break
            if args.terminal_mode == "angle":
                # Templates from THIS episode's own (upright) first frame, so the
                # detector is anchored at the episode's actual T position.
                templates, tc = make_templates(frame0_u8[b])
                tf = find_terminal_frame(imgs[b], templates, tc)
                if tf is None or tf + 1 < args.min_len:
                    if args.save_rejects_dir:
                        import cv2
                        import imageio.v2 as imageio
                        Path(args.save_rejects_dir).mkdir(parents=True, exist_ok=True)
                        a_last = est_angle(imgs[b, -1], templates, tc)
                        deg = "nan" if a_last is None else f"{np.degrees(a_last):+.0f}"
                        reason = "short" if tf is not None else "nocross"
                        up = np.stack([
                            cv2.resize(f, (384, 384), interpolation=cv2.INTER_NEAREST)
                            for f in imgs[b]
                        ])
                        imageio.mimwrite(
                            f"{args.save_rejects_dir}/reject{n_attempted:03d}_{reason}_"
                            f"final{deg}deg.mp4", up, fps=30, codec="libx264",
                            pixelformat="yuv420p", output_params=["-crf", "22"])
                    continue  # never reached ~80 deg CW (or degenerate) -> discard
                end = tf + 1
            else:
                end = args.ep_len
            rb.add_episode({
                "img": imgs[b, :end], "state": states[b, :end], "action": actions[b, :end],
            })
            ep_lens.append(end)
            n_done += 1
            n_acc += 1
            if n_videos_written < args.n_videos and args.video_dir:
                import cv2
                import imageio.v2 as imageio
                up = np.stack([
                    cv2.resize(f, (512, 512), interpolation=cv2.INTER_NEAREST)
                    for f in imgs[b, :end]
                ])
                imageio.mimwrite(
                    f"{args.video_dir}/imagined_ep{n_done - 1}_len{end}.mp4", up, fps=30,
                    codec="libx264", pixelformat="yuv420p", output_params=["-crf", "18"])
                n_videos_written += 1
        print(f"  batch {batch_i}: accepted {n_acc}/{bsz} in {time.time()-t0:.0f}s "
              f"({n_done}/{args.n_episodes} total, accept rate "
              f"{n_done / max(n_attempted, 1):.2f})", flush=True)
        batch_i += 1

    rb.save_to_path(args.out_zarr, if_exists="replace")
    ep_lens = np.array(ep_lens)
    print(f"Wrote {rb.n_episodes} imagined episodes / {rb.n_steps} steps to {args.out_zarr}")
    print(f"  episode lengths: min={ep_lens.min()} median={int(np.median(ep_lens))} "
          f"max={ep_lens.max()}  (real demos: 240-600)")
    print(f"  accept rate: {n_done}/{n_attempted} = {n_done / n_attempted:.2f}")


if __name__ == "__main__":
    main()

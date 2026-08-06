"""Evaluate the visual/dynamics quality of IWS world-model imagined rollouts
against the ground-truth MuJoCo simulator.

Three modes (--mode all runs everything):

rollout  PAIRED OPEN-LOOP REPLAY. Sample N initial states (random-init protocol:
         stabilize + arm settle + uprightness guard), snapshot the MuJoCo physics
         state, roll the DP closed-loop INSIDE the world model (exactly like
         collect_imagined_rotate_t), then restore each snapshot and replay the
         identical commanded action sequence through the real simulator +
         PID/IK controller. Frame k of the imagined rollout is compared to frame
         k of the sim replay: PSNR / SSIM / LPIPS vs step, plus the T rotation
         angle (sim = ground-truth physics state, WM = red-mask template
         estimator) and the success-label agreement (did the T cross 80 deg CW
         in the WM vs in the sim under the same actions?).

teacher  TEACHER-FORCED PREDICTION. Take real episodes from the rand800 pool,
         feed the WM each episode's real first frame + the real recorded action
         sequence, and compare predicted frames to the real stored frames.
         Isolates WM prediction quality from the DP's action distribution
         (rollout-mode actions may be OOD for the WM).

fid      DISTRIBUTIONAL METRICS. FID + KID (InceptionV3 pool3 features) between
         frames sampled from the imagined pools (randA+B+C, the actual training
         data of the scaling-study mixes) and frames from the real rand800 pool.
         A real-vs-real split (episodes 0-399 vs 400-799) is reported as the
         same-distribution floor.

Usage:
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python scripts/eval_wm_quality.py \
      --mode all --out_dir outputs/wm_quality
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "data_collection"))

import eval_dp_rotate_t as E  # noqa: E402
import eval_metric_correlation as M  # noqa: E402
import collect_rotate_t as C  # noqa: E402
import collect_imagined_rotate_t as CI  # noqa: E402
import gym_aloha.env as gae  # noqa: E402
from gym_aloha.env import AlohaEnv  # noqa: E402
from yixuan_utilities.kinematics_helper import KinHelper  # noqa: E402
from sim_aloha_dataset_collection_scripted import trajectory_to_joint_actions  # noqa: E402
from interactive_world_sim.utils.pose_utils import PoseType, pose_convert  # noqa: E402

DP_CKPT = ("/home/jacobhb/projects/worth_doing/diffusion_policy/data/outputs/"
           "2026.07.24/00.23.18_train_diffusion_unet_hybrid_rotate_t_image_eval/"
           "checkpoints/epoch=0010-test_mean_score=0.417.ckpt")  # randA/B/C collector
WM_CKPT = "ckpts/push_t/epoch=3-step=90000.ckpt"
REAL_ZARR = "datasets/rotate_t_rand800_dp.zarr"
IMAG_ZARRS = ["datasets/rotate_t_imagined_randA_dp.zarr",
              "datasets/rotate_t_imagined_randB_dp.zarr",
              "datasets/rotate_t_imagined_randC_dp.zarr"]
TERMINAL_RAD = np.radians(CI.TERMINAL_DEG)  # -80 deg (CW negative)


# ----------------------------------------------------------------------------- #
# frame metrics
# ----------------------------------------------------------------------------- #
def psnr_series(a, b):
    """a,b uint8 (T,H,W,3) -> (T,) PSNR in dB."""
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean(axis=(1, 2, 3))
    return 10 * np.log10(255.0**2 / np.maximum(mse, 1e-12))


def ssim_series(a, b):
    from skimage.metrics import structural_similarity as ssim
    return np.array([ssim(x, y, channel_axis=2, data_range=255)
                     for x, y in zip(a, b)])


def lpips_series(net, a, b, device, bs=64):
    out = []
    for i in range(0, len(a), bs):
        xa = torch.from_numpy(a[i:i + bs]).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1
        xb = torch.from_numpy(b[i:i + bs]).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1
        with torch.no_grad():
            out.append(net(xa, xb).flatten().cpu().numpy())
    return np.concatenate(out)


def angle_series(frames, templates, tc):
    """Estimated T angle (rad) per frame via the collector's estimator; nan if no T."""
    out = np.full(len(frames), np.nan)
    for k, f in enumerate(frames):
        a = CI.est_angle(f, templates, tc)
        if a is not None:
            out[k] = a
    return out


# ----------------------------------------------------------------------------- #
# rollout mode: paired DP-in-WM rollout vs sim replay of the same actions
# ----------------------------------------------------------------------------- #
def sample_initial_state(env, kin, seed, random_init):
    """Random-init protocol reset; returns obs + a restorable physics snapshot."""
    while True:
        np.random.seed(seed)
        env.reset(seed=seed)
        seed += 1
        C.stabilize_t(env)
        o = env._env.task.get_observation(env._env.physics)
        lb = pose_convert(o["left_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
        rb = pose_convert(o["right_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
        world_t_bases = np.stack([lb, rb])
        if random_init:
            C.settle_arms(env, world_t_bases, kin, np.zeros(6),
                          E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM)
            if abs(C.t_angle(env)) > C.UPRIGHT_TOL:
                continue
        image, agent_pos, frame_u8 = E.get_obs(env, kin, world_t_bases)
        snap = env._env.physics.get_state().copy()
        return image, agent_pos, frame_u8, world_t_bases, snap, seed


def sim_replay(env, kin, snap, world_t_bases, actions, frame0_u8):
    """Restore the snapshot and execute `actions` (T,4 EE-xy targets) through the
    standard controller. Returns (frames uint8 (T,128,128,3), gt_angles (T,))."""
    env.reset(seed=0)
    env._env.physics.set_state(snap)
    env._env.physics.forward()
    _, _, fr = E.get_obs(env, kin, world_t_bases)
    p0 = psnr_series(fr[None], frame0_u8[None])[0]
    if p0 < 40:
        print(f"    WARNING: snapshot restore imperfect (frame PSNR {p0:.1f} dB)")
    frames = [frame0_u8]
    gt_angles = [C.t_angle(env)]
    curr_vel = np.zeros(6)
    for target_xy in actions[:-1]:  # action[k] produces frame k+1
        o = env._env.task.get_observation(env._env.physics)
        joint, _ = trajectory_to_joint_actions(
            target_xy.astype(np.float64), world_t_bases, kin,
            o["qpos"][:14], curr_vel, E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM)
        env.step(joint)
        _, _, fr = E.get_obs(env, kin, world_t_bases)
        frames.append(fr)
        gt_angles.append(C.t_angle(env))
    return np.stack(frames), np.array(gt_angles)


def run_rollout(args, device, out_dir):
    print(f"Loading world model {args.wm_ckpt} ...")
    cfg = M.load_viz_cfg("pusht_mujoco")
    wm = M.load_model(args.wm_ckpt, cfg.algorithm, str(device))
    wm.dec_infer_steps = 1
    print(f"Loading DP policy {args.dp_ckpt} ...")
    policy, n_obs, n_act = E.load_policy(args.dp_ckpt, device)
    assert n_obs == 2
    import lpips
    lp = lpips.LPIPS(net="alex").to(device)

    env = AlohaEnv("pusht")
    kin = KinHelper(robot_name="trossen_vx300s")
    rng_xy = (-0.08, 0.08) if args.random_init else (0.0, 0.0)
    gae.sample_pusht_pose = C.make_upright_pose_sampler(rng_xy, rng_xy)

    B, L = args.n_episodes, args.ep_len
    torch.manual_seed(0)
    seed = args.env_seed_base
    inits = []
    for b in range(B):
        image, pos, frame, wtb, snap, seed = sample_initial_state(
            env, kin, seed, args.random_init)
        inits.append((image, pos, frame, wtb, snap))
    img0 = np.stack([i[0] for i in inits])
    home = np.stack([i[1] for i in inits]).astype(np.float32)
    frame0 = np.stack([i[2] for i in inits])

    print(f"Imagining {B} episodes x {L} frames ...")
    wm_imgs, _, actions = CI.imagine_batch(
        wm, policy, img0, home, frame0, B, L, n_act, device)

    psnr = np.empty((B, L))
    ssim = np.empty((B, L))
    lpip = np.empty((B, L))
    wm_ang = np.empty((B, L))
    gt_ang = np.empty((B, L))
    wm_crossed = np.zeros(B, bool)
    sim_crossed = np.zeros(B, bool)
    sim_all = np.empty((B, L, 128, 128, 3), np.uint8)
    vids = []
    for b in range(B):
        _, _, frame, wtb, snap = inits[b]
        sim_frames, gt = sim_replay(env, kin, snap, wtb, actions[b], frame)
        sim_all[b] = sim_frames
        psnr[b] = psnr_series(wm_imgs[b], sim_frames)
        ssim[b] = ssim_series(wm_imgs[b], sim_frames)
        lpip[b] = lpips_series(lp, wm_imgs[b], sim_frames, device)
        templates, tc = CI.make_templates(frame)
        wm_ang[b] = angle_series(wm_imgs[b], templates, tc)
        gt_ang[b] = gt
        wm_crossed[b] = CI.find_terminal_frame(wm_imgs[b], templates, tc) is not None
        sim_crossed[b] = bool((gt <= TERMINAL_RAD).any())
        print(f"  ep {b:02d}: PSNR@10/50/{L - 1} = {psnr[b, 10]:.1f}/{psnr[b, 50]:.1f}/"
              f"{psnr[b, -1]:.1f} dB  WM crossed={wm_crossed[b]} sim crossed={sim_crossed[b]}",
              flush=True)
        if len(vids) < args.n_videos:
            vids.append((wm_imgs[b], sim_frames))

    np.savez(out_dir / "rollout_metrics.npz", psnr=psnr, ssim=ssim, lpips=lpip,
             wm_angle=wm_ang, gt_angle=gt_ang, wm_crossed=wm_crossed,
             sim_crossed=sim_crossed, actions=actions)
    np.savez(out_dir / "rollout_frames.npz", wm=wm_imgs, sim=sim_all)

    if vids:
        import cv2
        import imageio.v2 as imageio
        vdir = out_dir / "videos"
        vdir.mkdir(exist_ok=True)
        for i, (wmf, simf) in enumerate(vids):
            up = np.stack([
                np.hstack([cv2.resize(a, (384, 384), interpolation=cv2.INTER_NEAREST),
                           cv2.resize(s, (384, 384), interpolation=cv2.INTER_NEAREST)])
                for a, s in zip(wmf, simf)])
            imageio.mimwrite(vdir / f"pair{i}_wm_vs_sim.mp4", up, fps=30,
                             codec="libx264", pixelformat="yuv420p",
                             output_params=["-crf", "20"])

    both = wm_crossed & sim_crossed
    print("\n===== ROLLOUT (paired open-loop replay) =====")
    print(f"episodes: {B} x {L} frames   (WM says success: {wm_crossed.sum()}, "
          f"sim confirms: {both.sum()}, sim-only: {(sim_crossed & ~wm_crossed).sum()})")
    for name, v, best in (("PSNR(dB)", psnr, "high"), ("SSIM", ssim, "high"),
                          ("LPIPS", lpip, "low")):
        print(f"  {name:9s} mean={v.mean():7.3f}  @t=10 {v[:, 10].mean():7.3f}  "
              f"@t=50 {v[:, 50].mean():7.3f}  @t={L - 1} {v[:, -1].mean():7.3f}")
    return dict(wm_success=int(wm_crossed.sum()), sim_confirms=int(both.sum()),
                n=B, psnr_mean=float(psnr.mean()), ssim_mean=float(ssim.mean()),
                lpips_mean=float(lpip.mean()))


# ----------------------------------------------------------------------------- #
# teacher mode: WM prediction on real episodes' actions vs the real frames
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def teacher_forced(wm, img0_f32, frame0_u8, actions, ep_len, n_act, device):
    """Roll the WM with GROUND-TRUTH actions. actions (B, ep_len+n_act, 4) raw.
    Returns predicted frames uint8 (B, ep_len, 128, 128, 3)."""
    from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
    B = img0_f32.shape[0]
    out = np.empty((B, ep_len, 128, 128, 3), np.uint8)
    out[:, 0] = frame0_u8
    acts = torch.from_numpy(actions).float()
    f0 = torch.from_numpy(img0_f32).to(device)
    z_hist = wm.encoder_forward(wm.normalizer["top_pov"].normalize(f0)).to(wm.dtype).unsqueeze(1)
    t = 0
    for _ in range(ep_len // n_act):
        lo = max(0, t - 9)
        act_win = wm.normalizer["action"].normalize(
            acts[:, lo:t + n_act + 1].to(device)).to(wm.dtype)
        z_new = wm.dynamics_forward(z_hist, act_win)
        z_hist = torch.cat([z_hist, z_new], dim=1)[:, -10:]
        dec = render_img_cm(
            wm, z_new.reshape(B * n_act, *z_new.shape[2:]), resolution=128,
            normalizer=wm.normalizer, num_views=1, batch_size=16,
        ).float().clamp(0, 1).reshape(B, n_act, 3, 128, 128)
        for j in range(n_act):
            k = t + 1 + j
            if k >= ep_len:
                break
            out[:, k] = (dec[:, j] * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        t += n_act
    return out


def run_teacher(args, device, out_dir):
    import zarr
    print(f"Loading world model {args.wm_ckpt} ...")
    cfg = M.load_viz_cfg("pusht_mujoco")
    wm = M.load_model(args.wm_ckpt, cfg.algorithm, str(device))
    wm.dec_infer_steps = 1
    import lpips
    lp = lpips.LPIPS(net="alex").to(device)

    B, L = args.n_episodes, args.ep_len
    z = zarr.open(args.real_zarr, mode="r")
    ends = z["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    lens = ends - starts
    n_act = 8
    eligible = np.nonzero(lens >= L + n_act + 1)[0]
    rng = np.random.default_rng(0)
    pick = rng.choice(eligible, size=B, replace=False)
    print(f"teacher-forcing {B} real episodes (of {len(eligible)} eligible >= "
          f"{L + n_act + 1} steps) from {args.real_zarr}")

    real = np.stack([z["data/img"][starts[e]:starts[e] + L] for e in pick])
    actions = np.stack([z["data/action"][starts[e]:starts[e] + L + n_act] for e in pick])
    img0 = np.moveaxis(real[:, 0], -1, 1).astype(np.float32) / 255.0

    pred = teacher_forced(wm, img0, real[:, 0], actions, L, n_act, device)

    psnr = np.stack([psnr_series(pred[b], real[b]) for b in range(B)])
    ssim = np.stack([ssim_series(pred[b], real[b]) for b in range(B)])
    lpip = np.stack([lpips_series(lp, pred[b], real[b], device) for b in range(B)])
    np.savez(out_dir / "teacher_metrics.npz", psnr=psnr, ssim=ssim, lpips=lpip,
             episodes=pick)
    np.savez(out_dir / "teacher_frames.npz", pred=pred, real=real)

    print("\n===== TEACHER-FORCED (real actions, real GT frames) =====")
    for name, v in (("PSNR(dB)", psnr), ("SSIM", ssim), ("LPIPS", lpip)):
        print(f"  {name:9s} mean={v.mean():7.3f}  @t=10 {v[:, 10].mean():7.3f}  "
              f"@t=50 {v[:, 50].mean():7.3f}  @t={L - 1} {v[:, -1].mean():7.3f}")
    return dict(psnr_mean=float(psnr.mean()), ssim_mean=float(ssim.mean()),
                lpips_mean=float(lpip.mean()))


# ----------------------------------------------------------------------------- #
# fid mode: FID + KID between imagined-pool frames and real-pool frames
# ----------------------------------------------------------------------------- #
def sample_frames(zarr_paths, n, rng, ep_range=None):
    """Uniformly sample n frames (uint8 HWC) across the given DP replay zarrs."""
    import zarr
    stores = []
    total = 0
    for p in zarr_paths:
        z = zarr.open(p, mode="r")
        ends = z["meta/episode_ends"][:]
        starts = np.concatenate([[0], ends[:-1]])
        if ep_range is not None:
            lo, hi = ep_range
            lo_s = starts[lo]
            hi_s = ends[min(hi, len(ends)) - 1]
        else:
            lo_s, hi_s = 0, ends[-1]
        stores.append((z["data/img"], lo_s, hi_s))
        total += hi_s - lo_s
    take = rng.choice(total, size=min(n, total), replace=False)
    take.sort()
    out = np.empty((len(take), 128, 128, 3), np.uint8)
    ofs = 0
    j = 0
    for img, lo_s, hi_s in stores:
        span = hi_s - lo_s
        sel = take[(take >= ofs) & (take < ofs + span)] - ofs + lo_s
        if len(sel):
            out[j:j + len(sel)] = img.get_orthogonal_selection((sel,))
            j += len(sel)
        ofs += span
    return out


def inception_acts(frames, device, bs=100):
    from pytorch_fid.inception import InceptionV3
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    acts = []
    with torch.no_grad():
        for i in range(0, len(frames), bs):
            x = torch.from_numpy(frames[i:i + bs]).permute(0, 3, 1, 2).float().to(device) / 255.0
            acts.append(model(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(acts)


def fid_from_acts(a1, a2):
    from pytorch_fid.fid_score import calculate_frechet_distance
    return float(calculate_frechet_distance(
        a1.mean(0), np.cov(a1, rowvar=False), a2.mean(0), np.cov(a2, rowvar=False)))


def kid_from_acts(a1, a2, n_subsets=100, subset_size=1000, seed=0):
    """Unbiased KID (MMD^2, poly kernel deg 3) x 1000, mean +/- std over subsets."""
    rng = np.random.default_rng(seed)
    d = a1.shape[1]
    m = min(subset_size, len(a1), len(a2))
    vals = []
    for _ in range(n_subsets):
        x = a1[rng.choice(len(a1), m, replace=False)]
        y = a2[rng.choice(len(a2), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        vals.append(((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                     + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                     - 2 * kxy.mean()) * 1000)
    return float(np.mean(vals)), float(np.std(vals))


def run_fid(args, device, out_dir):
    rng = np.random.default_rng(0)
    n = args.fid_samples
    print(f"Sampling {n} frames each: imagined pools / real pool / real halves ...")
    f_imag = sample_frames(args.imag_zarrs, n, rng)
    f_real = sample_frames([args.real_zarr], n, rng)
    f_ra = sample_frames([args.real_zarr], n, rng, ep_range=(0, 400))
    f_rb = sample_frames([args.real_zarr], n, rng, ep_range=(400, 800))
    print("Computing InceptionV3 activations ...")
    a_imag, a_real = inception_acts(f_imag, device), inception_acts(f_real, device)
    a_ra, a_rb = inception_acts(f_ra, device), inception_acts(f_rb, device)

    res = {
        "n_samples": int(len(f_imag)),
        "fid_imagined_vs_real": fid_from_acts(a_imag, a_real),
        "fid_real_floor": fid_from_acts(a_ra, a_rb),
    }
    res["kid_imagined_vs_real"], res["kid_imagined_vs_real_std"] = kid_from_acts(a_imag, a_real)
    res["kid_real_floor"], res["kid_real_floor_std"] = kid_from_acts(a_ra, a_rb)
    with open(out_dir / "fid.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\n===== DISTRIBUTIONAL (InceptionV3 pool3) =====")
    print(f"  FID  imagined vs real: {res['fid_imagined_vs_real']:.2f}   "
          f"(real-vs-real floor: {res['fid_real_floor']:.2f})")
    print(f"  KIDx1000 imagined vs real: {res['kid_imagined_vs_real']:.2f} "
          f"+/- {res['kid_imagined_vs_real_std']:.2f}   "
          f"(floor: {res['kid_real_floor']:.2f} +/- {res['kid_real_floor_std']:.2f})")
    return res


def run_fid_paired(args, device, out_dir):
    """FID/KID for the per-mode frame sets (frame 0 excluded — identical by
    construction): rollout WM vs its sim replay, teacher-forced predictions vs
    their real frames, sim-replay vs the real pool (control: should be ~floor),
    and a real-vs-real floor at the same sample size."""
    ro = np.load(out_dir / "rollout_frames.npz")
    te = np.load(out_dir / "teacher_frames.npz")
    w = ro["wm"][:, 1:].reshape(-1, 128, 128, 3)
    s = ro["sim"][:, 1:].reshape(-1, 128, 128, 3)
    p = te["pred"][:, 1:].reshape(-1, 128, 128, 3)
    r = te["real"][:, 1:].reshape(-1, 128, 128, 3)
    n = len(w)
    rng = np.random.default_rng(1)
    f_ra = sample_frames([args.real_zarr], n, rng, ep_range=(0, 400))
    f_rb = sample_frames([args.real_zarr], n, rng, ep_range=(400, 800))
    print(f"Computing InceptionV3 activations for paired sets ({n} frames each) ...")
    a = {k: inception_acts(v, device)
         for k, v in [("wm", w), ("sim", s), ("pred", p), ("real", r),
                      ("ra", f_ra), ("rb", f_rb)]}
    res = {"n_samples": int(n)}
    for name, x, y in [("rollout_wm_vs_simreplay", "wm", "sim"),
                       ("teacher_pred_vs_real", "pred", "real"),
                       ("simreplay_vs_realpool", "sim", "ra"),
                       ("real_floor_matched_n", "ra", "rb")]:
        res[f"fid_{name}"] = fid_from_acts(a[x], a[y])
        res[f"kid_{name}"], res[f"kid_{name}_std"] = kid_from_acts(a[x], a[y])
    with open(out_dir / "fid_paired.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\n===== PAIRED-SET DISTANCES (InceptionV3, frame 0 excluded) =====")
    for name in ("rollout_wm_vs_simreplay", "teacher_pred_vs_real",
                 "simreplay_vs_realpool", "real_floor_matched_n"):
        print(f"  {name:26s} FID={res[f'fid_{name}']:7.2f}   "
              f"KIDx1000={res[f'kid_{name}']:7.2f} +/- {res[f'kid_{name}_std']:.2f}")
    return res


# ----------------------------------------------------------------------------- #
# video mode: DINO subject consistency + RAFT optical-flow end-point error
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def dino_subject_consistency(videos, device, bs=256):
    """VBench-style subject consistency. videos (B,L,H,W,3) uint8 ->  (B,) scores:
    mean over frames of (cos(f_first, f_t) + cos(f_{t-1}, f_t)) / 2 on DINO
    ViT-S/16 features."""
    import torch.nn.functional as F
    model = torch.hub.load("facebookresearch/dino:main", "dino_vits16").to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    B, L = videos.shape[:2]
    flat = videos.reshape(B * L, *videos.shape[2:])
    feats = []
    for i in range(0, len(flat), bs):
        x = torch.from_numpy(flat[i:i + bs]).permute(0, 3, 1, 2).float().to(device) / 255.0
        x = (F.interpolate(x, size=224, mode="bilinear", align_corners=False) - mean) / std
        feats.append(F.normalize(model(x), dim=1).cpu())
    f = torch.cat(feats).reshape(B, L, -1)
    first = (f[:, :1] * f[:, 1:]).sum(-1)   # (B, L-1) cos to first frame
    adj = (f[:, :-1] * f[:, 1:]).sum(-1)    # (B, L-1) cos to previous frame
    return ((first + adj) / 2).mean(1).numpy()


def flow_color(flow):
    """Per-frame Middlebury color-wheel encoding, RGB in [0,1]. flow (N,2,H,W)."""
    from torchvision.utils import flow_to_image
    mx = flow.norm(dim=1).amax(dim=(1, 2)).clamp(min=1e-8).view(-1, 1, 1, 1)
    return flow_to_image(flow / mx).float() / 255.0


@torch.no_grad()
def flow_epe(vid_a, vid_b, device, bs=48):
    """Optical-flow end-point error between two PAIRED videos, two variants.
    vid_a/vid_b (B,L,H,W,3) uint8; RAFT flow on consecutive frames of each video.
    raw:   EPE(t) = mean_px || flow_a(t) - flow_b(t) ||_2  (flow space, px)
    color: AEPE(t) = mean_px l2 distance of the per-frame-normalized Middlebury
           color-wheel RGB encodings (the visualization-space protocol).
    Returns (epe_raw (B,L-1), mean |flow_b| (B,L-1), aepe_color (B,L-1),
    demo (flow-vis videos of episode 0: (L-1,H,2W,3) u8))."""
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    w = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=w).to(device).eval()
    tfm = w.transforms()

    def flows(video):
        t1 = torch.from_numpy(video[:-1]).permute(0, 3, 1, 2)
        t2 = torch.from_numpy(video[1:]).permute(0, 3, 1, 2)
        out = []
        for i in range(0, len(t1), bs):
            a, b = tfm(t1[i:i + bs].to(device), t2[i:i + bs].to(device))
            out.append(model(a, b)[-1].cpu())
        return torch.cat(out)  # (L-1, 2, H, W)

    B = vid_a.shape[0]
    epe = np.empty((B, vid_a.shape[1] - 1))
    mag = np.empty_like(epe)
    aepe = np.empty_like(epe)
    demo = None
    for b in range(B):
        fa, fb = flows(vid_a[b]), flows(vid_b[b])
        epe[b] = (fa - fb).norm(dim=1).mean((1, 2)).numpy()
        mag[b] = fb.norm(dim=1).mean((1, 2)).numpy()
        ca, cb = flow_color(fa), flow_color(fb)
        aepe[b] = (ca - cb).norm(dim=1).mean((1, 2)).numpy()
        if b == 0:
            demo = (torch.cat([ca, cb], dim=3).permute(0, 2, 3, 1)
                    * 255).byte().numpy()
        print(f"  ep {b:02d}: raw EPE {epe[b].mean():.3f} px  "
              f"color AEPE {aepe[b].mean():.4f}  "
              f"(GT flow magnitude {mag[b].mean():.3f} px)", flush=True)
    return epe, mag, aepe, demo


def run_video(args, device, out_dir):
    ro = np.load(out_dir / "rollout_frames.npz")
    te = np.load(out_dir / "teacher_frames.npz")
    print("DINO subject consistency (VBench protocol) ...")
    sc = {name: dino_subject_consistency(v, device)
          for name, v in [("wm_rollout", ro["wm"]), ("sim_replay", ro["sim"]),
                          ("teacher_pred", te["pred"]), ("real", te["real"])]}
    print("RAFT optical-flow EPE: rollout WM vs sim replay ...")
    epe_ro, mag_ro, aepe_ro, demo_ro = flow_epe(ro["wm"], ro["sim"], device)
    print("RAFT optical-flow EPE: teacher-forced vs real ...")
    epe_te, mag_te, aepe_te, _ = flow_epe(te["pred"], te["real"], device)

    np.savez(out_dir / "video_metrics.npz", epe_rollout=epe_ro, epe_teacher=epe_te,
             gtmag_rollout=mag_ro, gtmag_teacher=mag_te,
             aepe_color_rollout=aepe_ro, aepe_color_teacher=aepe_te,
             **{f"sc_{k}": v for k, v in sc.items()})
    if demo_ro is not None:
        import cv2
        import imageio.v2 as imageio
        up = np.stack([cv2.resize(f, (768, 384), interpolation=cv2.INTER_NEAREST)
                       for f in demo_ro])
        imageio.mimwrite(out_dir / "videos" / "pair0_flowvis_wm_vs_sim.mp4", up,
                         fps=30, codec="libx264", pixelformat="yuv420p",
                         output_params=["-crf", "20"])
    res = {f"subject_consistency_{k}": float(v.mean()) for k, v in sc.items()}
    res.update(epe_rollout_mean=float(epe_ro.mean()), epe_teacher_mean=float(epe_te.mean()),
               gt_flow_mag_rollout=float(mag_ro.mean()), gt_flow_mag_teacher=float(mag_te.mean()),
               aepe_color_rollout_mean=float(aepe_ro.mean()),
               aepe_color_teacher_mean=float(aepe_te.mean()))
    with open(out_dir / "video_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\n===== VIDEO METRICS =====")
    print("  subject consistency (DINO, higher=better):")
    for k, v in sc.items():
        print(f"    {k:13s} {v.mean():.4f} +/- {v.std():.4f}")
    print(f"  flow EPE rollout WM vs sim: {epe_ro.mean():.3f} px "
          f"(GT motion {mag_ro.mean():.3f} px/frame)   "
          f"color-space AEPE {aepe_ro.mean():.4f}")
    print(f"  flow EPE teacher vs real:   {epe_te.mean():.3f} px "
          f"(GT motion {mag_te.mean():.3f} px/frame)   "
          f"color-space AEPE {aepe_te.mean():.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "rollout", "teacher", "fid", "fid_paired",
                                       "video"],
                    default="all")
    ap.add_argument("--dp_ckpt", default=DP_CKPT)
    ap.add_argument("--wm_ckpt", default=WM_CKPT)
    ap.add_argument("--real_zarr", default=REAL_ZARR)
    ap.add_argument("--imag_zarrs", nargs="+", default=IMAG_ZARRS)
    ap.add_argument("--n_episodes", type=int, default=24)
    ap.add_argument("--ep_len", type=int, default=200, help="multiple of 8")
    ap.add_argument("--n_videos", type=int, default=4)
    ap.add_argument("--fid_samples", type=int, default=10000)
    ap.add_argument("--env_seed_base", type=int, default=900_000,
                    help="disjoint from collection (500k) and tight eval (7000)")
    ap.add_argument("--fixed_init", action="store_true")
    ap.add_argument("--out_dir", default="outputs/wm_quality")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    args.random_init = not args.fixed_init
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    if args.mode in ("all", "rollout"):
        summary["rollout"] = run_rollout(args, device, out_dir)
    if args.mode in ("all", "teacher"):
        summary["teacher"] = run_teacher(args, device, out_dir)
    if args.mode in ("all", "fid"):
        summary["fid"] = run_fid(args, device, out_dir)
    if args.mode in ("all", "fid_paired"):
        summary["fid_paired"] = run_fid_paired(args, device, out_dir)
    if args.mode in ("all", "video"):
        summary["video"] = run_video(args, device, out_dir)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll results in {out_dir}/")


if __name__ == "__main__":
    main()

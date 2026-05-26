#!/usr/bin/env python3
"""Visualize world model rollouts: predicted vs actual frames.

Loads a stage-2 dynamics checkpoint, seeds the rollout from the first frame
of each episode, replays ground-truth actions, and saves side-by-side
(predicted | ground-truth) MP4 videos.

Usage:
    conda activate iws-5090
    python scripts/visualize_rollout.py                          # defaults (pusht_mujoco)
    python scripts/visualize_rollout.py --config pusht_mujoco --n_episodes 5 --dec_infer_steps 1,3
"""
import argparse
import random
from pathlib import Path

import cv2
import h5py
import imageio
import numpy as np
import torch
from einops import rearrange
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)

# Match main.py / deploy/server.py: register resolvers so the visualization
# configs can use ${torch:bfloat16} etc. for dtype interpolations.
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
if not OmegaConf.has_resolver("torch"):
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))

CONFIG_DIR = Path(__file__).parents[1] / "configurations" / "visualization"


def load_viz_cfg(config_name: str) -> DictConfig:
    """Load a visualization config from configurations/visualization/<config_name>.yaml."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve()), version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


def load_model(ckpt_path: str, algo_cfg: DictConfig, device: str) -> LatentWorldModel:
    """Load a stage-2 checkpoint, mirroring deploy/server.py:load_model.

    Reads ``algo_cfg.dtype`` (default fp32) and casts the dynamics submodule to
    that dtype so SDPA can use the flash / mem-efficient kernel. The encoder
    stays in fp32 since image inputs arrive as fp32; the decoder is built at
    ``self.dtype`` inside ``LatentWorldModel._build_model``.
    """
    dtype = torch.float32 if "dtype" not in algo_cfg else algo_cfg.dtype
    algo = LatentWorldModel.load_from_checkpoint(
        ckpt_path,
        cfg=algo_cfg,
        strict=False,
        weights_only=False,
        map_location=device,
        dtype=dtype,
    )
    algo.dynamics = algo.dynamics.to(dtype)
    algo.eval()
    algo.dynamics.eval()
    return algo.to(device)


@torch.no_grad()
def rollout_episode(
    model: LatentWorldModel,
    episode_path: Path,
    t_pred: int,
    dec_infer_steps_list: list[int],
    device: str,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Run a world-model rollout for one episode with multiple decoder step counts.

    Seeds from frame 0, replays ground-truth actions for t_pred steps.

    Returns:
        all_pred_frames: list of (T, H, W, 3) uint8 arrays, one per dec_infer_steps value
        gt_frames:       (T, H, W, 3) uint8 — ground-truth frames 1..T
    """
    with h5py.File(episode_path, "r") as f:
        imgs = f["obs/images/top_pov"][:]  # (T, 128, 128, 3) uint8
        actions = f["action"][:]  # (T, 4) float32

    T_avail = len(imgs) - 1  # frames we can predict (need frame 0 as seed)
    T = min(t_pred, T_avail)

    # Prepare raw frames as (T+1, 3, 128, 128) float32 [0,1]
    frames_np = imgs[: T + 1].astype(np.float32) / 255.0  # (T+1, H, W, 3)
    frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).to(device)  # (T+1, 3, H, W)

    # Normalize observations and actions with the model's learned stats
    frames_norm = model.normalizer["top_pov"].normalize(frames)  # (T+1, 3, H, W)
    action_tensor = torch.from_numpy(actions[: T + 1]).float().to(device)
    action_norm = model.normalizer["action"].normalize(action_tensor)  # (T+1, 4)

    # Encode the first frame → initial latent z_0
    # Encoder is built in fp32, so feed it fp32 frames; the dynamics module is
    # cast to model.dtype (bf16 by default — see configurations/visualization/),
    # so we cast its inputs to match. Mirrors teleoperate_*.py.
    z_0 = model.encoder_forward(frames_norm[:1])  # (1, 4, 32, 32) fp32
    z_0_hist = z_0.unsqueeze(0).to(model.dtype)  # (B=1, T_hist=1, 4, 32, 32)

    # action input: (B=1, T_hist + T_act, 4) = (1, 1+T, 4)
    action_input = action_norm.unsqueeze(0).to(model.dtype)  # (1, T+1, 4)

    # Roll out dynamics once to get predicted latents for frames 1..T
    z_pred = model.dynamics_forward(z_0_hist, action_input)  # (1, T, 4, 32, 32)
    z_pred_flat = rearrange(z_pred, "b t c h w -> (b t) c h w")  # (T, 4, 32, 32)

    # Decode with each requested number of diffusion steps
    all_pred_frames = []
    for steps in dec_infer_steps_list:
        model.dec_infer_steps = steps
        pred_imgs = render_img_cm(
            model,
            z_pred_flat,
            resolution=128,
            normalizer=model.normalizer,
            num_views=1,
            batch_size=8,
        )  # (T, 3, 128, 128) in model.dtype, [0,1]
        # .float() before .numpy() because numpy doesn't support bfloat16
        # (mirrors deploy/server.py:383).
        pred_np = (pred_imgs.detach().cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
        pred_np = pred_np.transpose(0, 2, 3, 1)  # (T, H, W, 3)
        all_pred_frames.append(pred_np)

    gt_np = imgs[1 : T + 1]  # (T, H, W, 3) uint8
    return all_pred_frames, gt_np


def make_side_by_side_video(
    all_pred_frames: list[np.ndarray],
    dec_steps_list: list[int],
    gt_frames: np.ndarray,
    out_path: Path,
    fps: int,
) -> None:
    """Write a side-by-side (pred_steps_1 | pred_steps_2 | ... | GT) MP4."""
    T = min(min(len(p) for p in all_pred_frames), len(gt_frames))
    font = cv2.FONT_HERSHEY_SIMPLEX
    frames = []
    for t in range(T):
        panels = []
        for pred_frames, steps in zip(all_pred_frames, dec_steps_list):
            panel = pred_frames[t].copy()
            cv2.putText(panel, f"Pred {steps}step t={t+1:03d}", (4, 14), font, 0.4, (0, 0, 0), 1)
            panels.append(panel)

        gt = gt_frames[t].copy()
        gt[[0, 1, 2, -3, -2, -1], :] = [255, 0, 0]
        gt[:, [0, 1, 2, -3, -2, -1]] = [255, 0, 0]
        cv2.putText(gt, f"GT t={t+1:03d}", (4, 14), font, 0.4, (0, 0, 0), 1)
        panels.append(gt)

        frames.append(np.concatenate(panels, axis=1))

    with imageio.get_writer(str(out_path), fps=fps, codec="libx264", pixelformat="yuv420p", quality=None, ffmpeg_params=["-crf", "18"]) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"  -> {out_path}  ({T} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="pusht_mujoco",
        help="Name of the visualization config under configurations/visualization/ (default: pusht_mujoco)",
    )
    # The following args override the values from the config file when provided.
    parser.add_argument("--ckpt", default=None, help="Override checkpoint path")
    parser.add_argument("--data_dir", default=None, help="Override data directory")
    parser.add_argument("--out_dir", default=None, help="Override output directory")
    parser.add_argument("--n_episodes", type=int, default=None)
    parser.add_argument("--t_pred", type=int, default=None)
    parser.add_argument(
        "--dec_infer_steps",
        type=lambda s: [int(x) for x in s.split(",")],
        default=None,
        metavar="STEPS",
        help="Comma-separated decoder step counts, e.g. '1,3' (overrides config)",
    )
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_viz_cfg(args.config)

    # CLI overrides take precedence over config file values
    ckpt_path    = args.ckpt          or cfg.ckpt_path
    data_dir     = args.data_dir      or cfg.data_dir
    out_dir_str  = args.out_dir       or cfg.out_dir
    n_episodes   = args.n_episodes    or cfg.n_episodes
    t_pred       = args.t_pred        or cfg.t_pred
    dec_steps    = args.dec_infer_steps or list(cfg.dec_infer_steps)
    fps          = args.fps           or cfg.fps
    seed         = args.seed          or cfg.seed
    device       = args.device        or cfg.device

    random.seed(seed)
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {ckpt_path} on {device} ...")
    model = load_model(ckpt_path, cfg.algorithm, device)

    episodes = sorted(Path(data_dir).glob("episode_*.hdf5"))
    if not episodes:
        raise FileNotFoundError(f"No HDF5 episodes found in {data_dir}")

    chosen = random.sample(episodes, min(n_episodes, len(episodes)))
    print(f"Rolling out {len(chosen)} episodes (t_pred={t_pred}, dec_steps={dec_steps}) ...")

    for ep_path in chosen:
        print(f"  {ep_path.name} ...")
        all_pred_frames, gt_frames = rollout_episode(
            model, ep_path, t_pred, dec_steps, device
        )
        make_side_by_side_video(
            all_pred_frames, dec_steps, gt_frames, out_dir / f"{ep_path.stem}.mp4", fps
        )

    print(f"\nDone. Videos saved to {out_dir}/")


if __name__ == "__main__":
    main()

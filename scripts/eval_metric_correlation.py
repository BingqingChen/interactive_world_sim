#!/usr/bin/env python3
"""Compute per-window world-model metrics for correlation analysis.

For each window sampled from a directory of validation episodes, this script
rolls out the latent world model ``horizon`` frames into the future from
``n_context`` ground-truth context frames (replaying ground-truth actions) and
writes one CSV row per window with:

- ``psnr_db``:            absolute PSNR (dB) of the decoded canonical rollout
                          vs the ground-truth frames (pixel space, [0, 1]).
- ``interseed_var``:      inter-seed variance of the predicted latents across
                          ``n_seeds`` independent stochastic rollouts.
- ``roundtrip_residual``: encoder round-trip residual of the predicted latents
                          (decode -> re-encode -> compare).

The metrics follow the mmbench2 "hallucination predictor" analysis (Figure 5)
but keep absolute values: no repeated-frame PSNR baseline and no scene-motion
normalization, since all windows come from a single task.

Plot the resulting CSV with scripts/plot_metric_correlation.py.

Usage:
    python scripts/eval_metric_correlation.py                  # defaults (push_t)
    python scripts/eval_metric_correlation.py --n_windows 5    # quick smoke test
"""
import argparse
import csv
import math
from pathlib import Path

import h5py
import numpy as np
import torch
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
    """Load a stage-2 checkpoint, mirroring scripts/visualize_rollout.py:load_model.

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


def sample_windows(
    episode_paths: list[Path],
    n_windows: int,
    window_len: int,
    rng: np.random.Generator,
) -> list[tuple[Path, int]]:
    """Sample (episode, start) windows uniformly over all valid positions.

    Every contiguous window of ``window_len`` frames across all episodes is a
    candidate; ``n_windows`` of them are drawn without replacement.
    """
    candidates: list[tuple[Path, int]] = []
    for path in episode_paths:
        with h5py.File(path, "r") as f:
            episode_len = f["action"].shape[0]
        for start in range(episode_len - window_len + 1):
            candidates.append((path, start))
    if not candidates:
        raise ValueError(
            f"No episode is long enough for a {window_len}-frame window"
        )
    n_windows = min(n_windows, len(candidates))
    chosen = rng.choice(len(candidates), size=n_windows, replace=False)
    return [candidates[i] for i in chosen]


@torch.no_grad()
def evaluate_window(
    model: LatentWorldModel,
    episode_path: Path,
    start: int,
    horizon: int,
    n_context: int,
    n_seeds: int,
    window_seed: int,
    resolution: int,
    device: str,
    dump_path: Path | None = None,
) -> dict:
    """Roll out one window with ``n_seeds`` stochastic samples and compute metrics.

    Returns a dict with the window identity and three scalars:

    - ``interseed_var``: variance across the ``n_seeds`` rollouts per latent
      element -> mean over latent dims (C, H, W) -> sqrt -> mean over the
      horizon. Same formula as mmbench2's inter-seed variance, except the N
      samples here are full independent rollouts (mmbench2 branches N samples
      per step from a shared past), and the value is kept absolute (no
      scene-motion normalization).
    - ``psnr_db``: absolute PSNR of the decoded canonical rollout (seed 0)
      against ground truth, MSE averaged over (horizon, C, H, W) in [0, 1].
    - ``roundtrip_residual``: decode the canonical latents, re-encode the
      decoded frames, and take the RMS latent difference per step -> mean over
      the horizon.
    """
    window_len = n_context + horizon
    with h5py.File(episode_path, "r") as f:
        imgs = f["obs/images/top_pov"][start : start + window_len]  # (L, H, W, 3) uint8
        actions = f["action"][start : start + window_len]  # (L, 4) float32

    # Normalize observations and actions with the model's learned stats.
    frames_np = imgs.astype(np.float32) / 255.0
    frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).to(device)  # (L, 3, H, W)
    frames_norm = model.normalizer["top_pov"].normalize(frames)
    action_tensor = torch.from_numpy(actions).float().to(device)
    action_norm = model.normalizer["action"].normalize(action_tensor)  # (L, 4)

    # Encode the context frames, then repeat everything n_seeds times along the
    # batch dim: dynamics_forward draws its noise independently per batch
    # element, so one batched call yields n_seeds independent rollouts.
    z_ctx = model.encoder_forward(frames_norm[:n_context])  # (n_context, C_l, H_l, W_l)
    z_hist = z_ctx.unsqueeze(0).repeat(n_seeds, 1, 1, 1, 1).to(model.dtype)
    action_input = action_norm.unsqueeze(0).repeat(n_seeds, 1, 1).to(model.dtype)

    torch.manual_seed(window_seed)  # also seeds CUDA; makes the window reproducible
    z_pred = model.dynamics_forward(z_hist, action_input)  # (n_seeds, horizon, C_l, H_l, W_l)

    interseed_var = (
        z_pred.float().var(dim=0, unbiased=False).mean(dim=(1, 2, 3)).sqrt().mean().item()
    )

    # Decode the canonical rollout (seed 0) once; it serves both PSNR and the
    # round-trip residual.
    z_canon = z_pred[0]  # (horizon, C_l, H_l, W_l)
    pred_imgs = render_img_cm(
        model,
        z_canon,
        resolution=resolution,
        normalizer=model.normalizer,
        num_views=len(model.obs_keys),
        batch_size=8,
    ).float()  # (horizon, 3, H, W) in [0, 1]

    gt_frames = frames[n_context:]  # (horizon, 3, H, W) in [0, 1]
    mse = (pred_imgs - gt_frames).pow(2).mean().item()
    psnr_db = 10.0 * math.log10(1.0 / max(mse, 1e-12))

    pred_norm = model.normalizer["top_pov"].normalize(pred_imgs)
    z_reenc = model.encoder_forward(pred_norm)  # (horizon, C_l, H_l, W_l)
    roundtrip_residual = (
        (z_canon.float() - z_reenc).pow(2).mean(dim=(1, 2, 3)).sqrt().mean().item()
    )

    # Optionally save the decoded imagined frames + GT frames + actions so a
    # second-stage script can score them with other world models (e.g. the JEPA
    # latent-consistency hallucination detector). Frames are uint8 HWC [0,255].
    if dump_path is not None:
        pred_uint8 = (
            (pred_imgs.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        )  # (horizon, H, W, 3)
        np.savez_compressed(
            dump_path,
            iws_imagined=pred_uint8,  # (horizon, H, W, 3) diffusion-model rollout
            gt_frames=imgs,  # (window_len, H, W, 3) ground truth (frame 0 = clean start)
            actions=actions,  # (window_len, 4)
            episode=episode_path.name,
            start=start,
            n_context=n_context,
            horizon=horizon,
        )

    return {
        "episode": episode_path.name,
        "start": start,
        "psnr_db": psnr_db,
        "interseed_var": interseed_var,
        "roundtrip_residual": roundtrip_residual,
        "horizon": horizon,
        "n_context": n_context,
        "n_seeds": n_seeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        default="pusht_mujoco",
        help="Visualization config under configurations/visualization/ supplying the "
        "algorithm block (default: pusht_mujoco)",
    )
    parser.add_argument("--ckpt", default="ckpts/push_t/epoch=3-step=90000.ckpt")
    parser.add_argument("--data_dir", default="datasets/val")
    parser.add_argument("--out_csv", default="outputs/metric_correlation/window_metrics.csv")
    parser.add_argument("--n_windows", type=int, default=200)
    parser.add_argument("--n_seeds", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=24, help="Predicted frames per window")
    parser.add_argument("--n_context", type=int, default=1, help="Ground-truth context frames")
    parser.add_argument("--dec_infer_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dump_frames_dir",
        default=None,
        help="If set, save per-window decoded imagined + GT frames and actions as "
        "<dir>/<episode>_<start>.npz for a second-stage scorer (e.g. JEPA consistency).",
    )
    args = parser.parse_args()

    cfg = load_viz_cfg(args.config)
    resolution = cfg.algorithm.x_shape[-1]

    print(f"Loading model from {args.ckpt} on {args.device} ...")
    model = load_model(args.ckpt, cfg.algorithm, args.device)
    model.dec_infer_steps = args.dec_infer_steps

    episodes = sorted(Path(args.data_dir).glob("episode_*.hdf5"))
    if not episodes:
        raise FileNotFoundError(f"No HDF5 episodes found in {args.data_dir}")

    rng = np.random.default_rng(args.seed)
    windows = sample_windows(episodes, args.n_windows, args.n_context + args.horizon, rng)
    print(
        f"Evaluating {len(windows)} windows from {len(episodes)} episodes "
        f"(n_context={args.n_context}, horizon={args.horizon}, n_seeds={args.n_seeds}) ..."
    )

    # Rows are written and flushed as they are computed, so a long run that is
    # interrupted still leaves a usable CSV of everything finished so far.
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dump_dir = Path(args.dump_frames_dir) if args.dump_frames_dir else None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
    n_done = 0
    with open(out_csv, "w", newline="") as f:
        writer = None
        for i, (episode_path, start) in enumerate(windows):
            dump_path = (
                dump_dir / f"{episode_path.stem}_{start}.npz" if dump_dir else None
            )
            row = evaluate_window(
                model,
                episode_path,
                start,
                horizon=args.horizon,
                n_context=args.n_context,
                n_seeds=args.n_seeds,
                window_seed=args.seed * 100_000 + i,
                resolution=resolution,
                device=args.device,
                dump_path=dump_path,
            )
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            n_done += 1
            print(
                f"  [{i + 1}/{len(windows)}] {row['episode']} start={row['start']:<3d} "
                f"psnr={row['psnr_db']:.2f}dB "
                f"interseed_var={row['interseed_var']:.4f} "
                f"roundtrip={row['roundtrip_residual']:.4f}"
            )

    print(f"\nWrote {n_done} rows to {out_csv}")
    print(f"Plot with: python scripts/plot_metric_correlation.py --csv {out_csv}")


if __name__ == "__main__":
    main()

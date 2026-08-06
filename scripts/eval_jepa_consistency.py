#!/usr/bin/env python3
"""Stage 2: JEPA/DINO latent-consistency hallucination scores for the diffusion rollout.

Reads the per-window frame dumps written by
``eval_metric_correlation.py --dump_frames_dir`` (the IWS diffusion model's decoded
imagined frames + ground-truth frames + actions) and, for each window, scores the
diffusion rollout with two post-trained JEPA world models (DINO-WM, V-JEPA-2) acting as a
hallucination detector. For a given JEPA model, at each rollout step t:

    e_F(t) = || z_hat_t - E(F_iws_t) ||_2

is the plain Euclidean norm over the flattened latent grid (V*16*16*D) between the JEPA
predictor's latent ``z_hat_t`` and the frozen encoder's latent of the diffusion model's
imagined frame at t. Matches intern_project ``distance(..., metric="l2")``. Two rollout
modes (they differ by exactly one line -- what gets fed back as context):

  - openloop:  JEPA rolls out its own latents from the clean GT start frame, feeding its
               own predictions back. e_F = drift of the diffusion rollout from JEPA's
               independent imagination.
  - reanchor:  JEPA re-conditions on the diffusion trajectory every step (feeds the
               imagined frame's latent back). e_F = per-step "does F_iws_t follow from
               F_iws_{<t} under JEPA dynamics" -- the classic consistency check.

Per window we store the horizon-mean and final-step e_F for each model x mode, merged
into the stage-1 CSV on (episode, start).

Usage:
    python scripts/eval_jepa_consistency.py \
        --frames_dir outputs/metric_correlation/frames \
        --in_csv  outputs/metric_correlation/window_metrics.csv \
        --out_csv outputs/metric_correlation/window_metrics_jepa.csv
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

JEPA_REPO = "/home/jacobhb/projects/worth_doing/jepa-wms"

# jepa-wms environment (mirrors iws_env.sh); force-set (not setdefault) because a shell
# profile may export stale values, and these must be correct before importing app.* and
# before the config's ${JEPAWM_OSSCKPT} is expanded (that path holds the V-JEPA-2 giant
# encoder, which is not stored in the fine-tune checkpoint).
os.environ["JEPAWM_HOME"] = JEPA_REPO
os.environ["JEPAWM_LOGS"] = f"{JEPA_REPO}/logs"
os.environ["JEPAWM_CKPT"] = f"{JEPA_REPO}/logs"
os.environ["JEPAWM_DSET"] = "/home/jacobhb/projects/worth_doing/interactive_world_sim"
os.environ["JEPAWM_OSSCKPT"] = "/home/jacobhb/projects/intern_project/ckpts"

# bimanual_push action dim (overrides consistency_scorer.build_video_wm's hardcoded libero 7).
IWS_ACTION_DIM = 4

MODELS = {
    "dino": {
        "config": f"{JEPA_REPO}/configs/vjepa_wm/iws_pusht/iws_pusht_h1_dino.yaml",
        "ckpt": f"{JEPA_REPO}/logs/iws_pusht/ft_h1_dino/jepa-latest.pth.tar",
    },
    "vjepa": {
        "config": f"{JEPA_REPO}/configs/vjepa_wm/iws_pusht/iws_pusht_h1_vjepa.yaml",
        "ckpt": f"{JEPA_REPO}/logs/iws_pusht/ft_h1_vjepa/jepa-latest.pth.tar",
    },
}


def build_iws_wm(config_path: str, ckpt_path: str, device: str):
    """Build a VideoWM for an iws_pusht fine-tune and load its checkpoint.

    Mirrors ``consistency_scorer.build_video_wm`` but (a) expands ``${ENV}`` in the config
    (so the V-JEPA encoder path resolves) and (b) uses the 4-d bimanual_push action dim
    instead of the hardcoded libero 7-d. Returns (wm, cfg, transform).
    """
    if JEPA_REPO not in sys.path:
        sys.path.insert(0, JEPA_REPO)
    from src.utils.yaml_utils import expand_env_vars
    from app.plan_common.datasets.transforms import make_transforms
    from app.vjepa_wm.utils import init_video_model, load_checkpoint
    from app.vjepa_wm.video_wm import VideoWM

    cfg = expand_env_vars(yaml.safe_load(open(config_path)))
    cm, cd, cc = cfg["model"], cfg["data"], cfg["data"]["custom"]
    img_size = cd["img_size"]
    frameskip, action_skip = cc["frameskip"], cc["action_skip"]
    tubelet_size_enc = cm["tubelet_size_enc"]
    action_tokens = cm["action_encoder"]["action_tokens"]
    proprio_tokens = cm["proprio_encoder"]["proprio_tokens"]
    action_emb_dim = cm["action_encoder"]["action_emb_dim"]
    proprio_emb_dim = cm["proprio_encoder"]["proprio_emb_dim"]
    use_proprio = proprio_tokens > 0 or proprio_emb_dim > 0
    use_action = action_tokens > 0 or action_emb_dim > 0
    model_action_dim = (
        IWS_ACTION_DIM * tubelet_size_enc * frameskip // action_skip if use_action else None
    )
    model_proprio_dim = 7 * tubelet_size_enc if use_proprio else None

    excluded = ["rollout_cfg", "heads_cfg", "pretrained_path", "visual_encoder",
                "action_encoder", "proprio_encoder", "predictor", "wm_encoding", "attn"]
    mk = {k: v for k, v in cm.items() if k not in excluded}
    mk.update(cm["visual_encoder"])
    mk.update(cm["action_encoder"])
    mk.update(cm["proprio_encoder"])
    mk.update(cm["predictor"])
    mk.update(dict(device=device, img_size=img_size, action_dim=model_action_dim,
                   proprio_dim=model_proprio_dim, cfgs_attn_pattern=cm["attn"],
                   use_proprio=use_proprio, use_action=use_action))
    predictor, encoder, action_encoder, proprio_encoder = init_video_model(**mk)
    predictor, action_encoder, proprio_encoder, _, _, _, _ = load_checkpoint(
        r_path=ckpt_path, predictor=predictor, action_encoder=action_encoder,
        proprio_encoder=proprio_encoder, heads={}, opt=None, scaler=None,
        load_opt_scale_epoch=False, load_heads=False)
    wm = VideoWM(
        device=device, encoder=encoder, predictor=predictor, action_encoder=action_encoder,
        proprio_encoder=proprio_encoder, action_dim=model_action_dim, proprio_dim=model_proprio_dim,
        use_proprio=use_proprio, use_action=use_action, action_tokens=action_tokens,
        proprio_tokens=proprio_tokens, grid_size=cm["grid_size"], tubelet_size_enc=tubelet_size_enc,
        action_conditioning=cm["action_conditioning"], proprio_encoding=cm["proprio_encoding"],
        enc_type=cm["visual_encoder"]["enc_type"], pred_type=cm["predictor"]["pred_type"],
        action_encoder_inpred=cm["action_encoder"]["action_encoder_inpred"],
        proprio_encoder_inpred=cm["proprio_encoder"]["proprio_encoder_inpred"],
        **cm["wm_encoding"], action_skip=action_skip, frameskip=frameskip, img_size=img_size,
        heads={}, scaler=None, optimizer=None, clip_grad=None, mixed_precision=False,
        use_radamw=False, cfgs_loss=cfg["loss"],
    ).to(device).eval()
    transform = make_transforms(img_size=img_size, **cfg["data_aug"])
    return wm, cfg, transform


@torch.no_grad()
def encode_frames(wm, transform, frames_uint8: np.ndarray, device: str) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> per-frame latents (N, V, H_l, W_l, D) via the frozen encoder."""
    x = torch.as_tensor(frames_uint8).float().permute(0, 3, 1, 2) / 255.0  # (N, 3, H, W) [0,1]
    x = transform(x)  # resize to img_size + per-model normalize
    x = x.unsqueeze(1).to(device)  # (N, 1, 3, img, img): batch=N frames, tau=1 each
    z = wm.encode_obs({"visual": x})["visual"]  # (N, 1, V, H_l, W_l, D)
    return z[:, 0]  # (N, V, H_l, W_l, D)


@torch.no_grad()
def rollout_consistency(wm, z0, z_iws, act_feats, ctxt_window, mode):
    """Per-step consistency error e_F(t) = ||z_hat_t - z_iws[t-1]|| for one rollout mode.

    Args:
        z0:        (V, H, W, D) clean-start latent (encoded GT frame 0).
        z_iws:     (horizon, V, H, W, D) latents of the diffusion imagined frames 1..horizon
                   (both the comparison targets and, in "reanchor" mode, the fed-back context).
        act_feats: encoded actions, dim-1 indexed by frame so act_feats[:, i] takes frame i->i+1
                   (dino: (1, horizon, grid^2, A); vjepa: (1, horizon, A)).
        mode:      "openloop" (feed JEPA's own prediction back) or "reanchor" (feed z_iws back).

    Returns: numpy (horizon,) of Euclidean norms.
    """
    horizon = z_iws.shape[0]
    ctx = [z0]           # list of (V, H, W, D), frames 0..t-1
    act_idx = []         # action index aligned to each context frame
    ef = np.empty(horizon, dtype=np.float64)
    for t in range(1, horizon + 1):
        act_idx.append(t - 1)  # action a_{t-1} drives frame t-1 -> t
        vid = torch.stack(ctx[-ctxt_window:], dim=0).unsqueeze(0)  # (1, k, V, H, W, D)
        afeat = act_feats[:, act_idx[-ctxt_window:]]               # (1, k, ...)
        pred, _, _ = wm.forward_pred(vid, afeat, None)
        z_hat = pred[:, -1][0]                                     # (V, H, W, D)
        z_ref = z_iws[t - 1]                                       # (V, H, W, D)
        ef[t - 1] = (z_hat.float() - z_ref.float()).flatten().norm().item()
        ctx.append(z_hat if mode == "openloop" else z_iws[t - 1])
    return ef


@torch.no_grad()
def score_window(wm, transform, npz, device, ctxt_window, use_bf16):
    """Return {mode: (ef_mean, ef_final)} for one window under this JEPA model."""
    gt = npz["gt_frames"]            # (window_len, H, W, 3) uint8
    iws = npz["iws_imagined"]        # (horizon, H, W, 3) uint8
    n_context = int(npz["n_context"])
    horizon = int(npz["horizon"])
    actions = torch.as_tensor(npz["actions"][:horizon]).float().unsqueeze(0).to(device)  # (1, horizon, 4)

    clean_start = gt[n_context - 1 : n_context]  # (1, H, W, 3) the frame the diffusion seeded from
    frames = np.concatenate([clean_start, iws], axis=0)  # (1 + horizon, H, W, 3)

    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)
    with ctx:
        zf = encode_frames(wm, transform, frames, device)  # (1+horizon, V, H, W, D)
        z0, z_iws = zf[0], zf[1:]
        act_feats = wm.encode_act(actions)  # dino: (1,horizon,grid^2,A); vjepa: (1,horizon,A)
        out = {}
        for mode in ("openloop", "reanchor"):
            ef = rollout_consistency(wm, z0, z_iws, act_feats, ctxt_window, mode)
            out[mode] = (float(ef.mean()), float(ef[-1]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--frames_dir", default="outputs/metric_correlation/frames")
    parser.add_argument("--in_csv", default="outputs/metric_correlation/window_metrics.csv")
    parser.add_argument("--out_csv", default="outputs/metric_correlation/window_metrics_jepa.csv")
    parser.add_argument("--models", default="dino,vjepa", help="Comma-separated subset of dino,vjepa")
    parser.add_argument("--ctxt_window", type=int, default=3, help="Sliding latent context (<=3)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N windows")
    parser.add_argument("--no_bf16", action="store_true", help="Encode/predict in fp32 instead of bf16 autocast")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    npz_files = sorted(frames_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz frame dumps in {frames_dir} (run stage 1 with --dump_frames_dir)")
    if args.limit:
        npz_files = npz_files[: args.limit]
    print(f"Scoring {len(npz_files)} windows with models: {args.models}")

    # rows keyed by (episode, start); one dict per window accumulating all model x mode columns
    rows: dict[tuple[str, int], dict] = {}
    for model_name in args.models.split(","):
        model_name = model_name.strip()
        m = MODELS[model_name]
        print(f"\nLoading {model_name} from {m['ckpt']} ...")
        wm, _, transform = build_iws_wm(m["config"], m["ckpt"], args.device)
        for i, npz_path in enumerate(npz_files):
            npz = np.load(npz_path, allow_pickle=True)
            key = (str(npz["episode"]), int(npz["start"]))
            scores = score_window(wm, transform, npz, args.device, args.ctxt_window, not args.no_bf16)
            row = rows.setdefault(key, {"episode": key[0], "start": key[1]})
            for mode, (ef_mean, ef_final) in scores.items():
                row[f"{model_name}_ef_{mode}_mean"] = ef_mean
                row[f"{model_name}_ef_{mode}_final"] = ef_final
            if (i + 1) % 50 == 0 or i + 1 == len(npz_files):
                print(f"  [{model_name}] {i + 1}/{len(npz_files)}")
        del wm
        torch.cuda.empty_cache()

    jepa_df = pd.DataFrame(list(rows.values()))
    base = pd.read_csv(args.in_csv)
    merged = base.merge(jepa_df, on=["episode", "start"], how="inner")
    # Guard against pairing e_F with the wrong PSNR: every scored window MUST match a row
    # in in_csv. A mismatch means in_csv is a different run than the one that produced these
    # frame dumps (e.g. the default window_metrics.csv vs the matching *_stage1.csv), which
    # would silently mispair PSNR. Fail loudly instead.
    if len(merged) != len(jepa_df):
        missing = len(jepa_df) - len(merged)
        raise ValueError(
            f"{missing}/{len(jepa_df)} scored windows have no matching (episode,start) row in "
            f"{args.in_csv}. The frame dumps in {args.frames_dir} were produced by a different "
            f"stage-1 run than this CSV -- pass the --in_csv that was written alongside "
            f"--dump_frames_dir so each window's e_F pairs with its own PSNR."
        )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"\nWrote {len(merged)} rows ({len(jepa_df.columns) - 2} JEPA columns) to {out_csv}")
    ef_cols = [c for c in merged.columns if "_ef_" in c]
    print("JEPA consistency columns:", ef_cols)


if __name__ == "__main__":
    main()

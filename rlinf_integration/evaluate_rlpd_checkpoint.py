"""Closed-loop success-rate evaluation of a trained RLPD-in-real-sim checkpoint,
following the canonical protocol of run_scaling_v3_sweep.py (N=50, seed 9000,
max_steps 800, x/y in +-0.06, feasible_mask_v3; success = eval_success: angle in
[-105,-80] deg AND flat).

Loads CNNPolicy weights in this process (RLinf's venv) and bridges env steps to
real_sim_chunk_server.py (IWS's venv), like IWSRotateTSimEnv. Actions are
deterministic (mode="eval": the tanh-squashed mean). One real episode per row;
each row's first completion is its result. Failures are split into
timeout / overshoot (< -105 deg) / undershoot (stopped short of -80 deg) /
tipped (not flat), from the server's per-episode report.

Usage:
  MUJOCO_GL=egl ~/RLinf/.venv/bin/python rlinf_integration/evaluate_rlpd_checkpoint.py \
      --checkpoint_dir .../checkpoints/global_step_1800/actor \
      --state_history none --action_mode delta --out_json eval.json
"""
import argparse
import json
import os
import pickle
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch

WORKTREE_ROOT = Path(__file__).resolve().parent.parent
IWS_VENV_PYTHON = "/home/jacobhb/projects/worth_doing/interactive_world_sim/.venv/bin/python"
RESNET_MODEL_PATH = "/home/jacobhb/projects/intern_project/RLinf/RLinf-ResNet10-pretrained"
STATE_DIMS = {"single": 4, "concat_state": 8, "none": 1}  # none: constant zero state

from rlinf.models.embodiment.cnn_policy.cnn_policy import CNNPolicy, CNNConfig  # noqa: E402


def load_policy(checkpoint_dir, state_dim, action_dim, device):
    cfg = CNNConfig()
    cfg.update_from_dict(dict(
        image_size=[3, 128, 128], image_num=1, action_dim=action_dim, state_dim=state_dim,
        num_action_chunks=1, backbone="resnet", model_path=RESNET_MODEL_PATH,
        encoder_config={"ckpt_name": "resnet10_pretrained.pt"},
        add_value_head=False, add_q_head=True, num_q_heads=10,
    ))
    model = CNNPolicy(cfg)
    sd = torch.load(Path(checkpoint_dir) / "model_state_dict" / "full_weights.pt",
                    map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: missing={missing} unexpected={unexpected}")
    return model.to(device).eval()


class ServerClient:
    def __init__(self, args, num_envs):
        server_script = WORKTREE_ROOT / "rlinf_integration" / "real_sim_chunk_server.py"
        self.proc = subprocess.Popen(
            [IWS_VENV_PYTHON, "-u", str(server_script),
             "--num_envs", str(num_envs), "--max_steps", str(args.max_steps),
             "--n_obs", str(args.n_obs),
             "--x_min", str(args.x_min), "--x_max", str(args.x_max),
             "--y_min", str(args.y_min), "--y_max", str(args.y_max),
             "--feasible_mask", str(args.feasible_mask),
             "--env_seed_base", str(args.eval_seed_base),
             "--action_mode", args.action_mode, "--max_step", str(args.max_step),
             "--iws_scripts_root", str(WORKTREE_ROOT)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            env={**os.environ, "MUJOCO_GL": "egl"},
        )

    def _send(self, req_byte, payload=None):
        self.proc.stdin.write(req_byte)
        if payload is not None:
            data = pickle.dumps(payload, protocol=4)
            self.proc.stdin.write(struct.pack(">Q", len(data)))
            self.proc.stdin.write(data)
        self.proc.stdin.flush()
        header = self.proc.stdout.read(8)
        if len(header) < 8:
            raise RuntimeError("real_sim_chunk_server closed its stdout unexpectedly")
        (length,) = struct.unpack(">Q", header)
        return pickle.loads(self.proc.stdout.read(length))

    def reset_all(self):
        return self._send(b"\x01")

    def step_chunk(self, actions):
        return self._send(b"\x02", actions)

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.terminate()


def wrap_obs(imgs, aps, state_history, device):
    main_images = torch.from_numpy(imgs[:, -1]).to(device).permute(0, 2, 3, 1) * 255.0
    if state_history == "single":
        states_np = aps[:, -1]
    elif state_history == "concat_state":
        states_np = np.concatenate([aps[:, -2], aps[:, -1]], axis=-1)
    else:
        states_np = np.zeros((aps.shape[0], 1), np.float32)  # image-only: constant zero state
    return {"main_images": main_images, "states": torch.from_numpy(states_np).to(device)}


def classify(r):
    if r["success"]:
        return "success"
    if not (r["init_flat"] and r["final_flat"]):
        return "tipped"
    if r["delta_deg"] < -105.0:
        return "overshoot"
    if r["timeout"]:
        return "timeout"
    return "undershoot"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--state_history", choices=list(STATE_DIMS), required=True)
    ap.add_argument("--action_mode", choices=["absolute", "delta"], default="absolute")
    ap.add_argument("--max_step", type=float, default=0.04)
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--eval_seed_base", type=int, default=9000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_act", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--n_obs", type=int, default=2)
    ap.add_argument("--x_min", type=float, default=-0.06)
    ap.add_argument("--x_max", type=float, default=0.06)
    ap.add_argument("--y_min", type=float, default=-0.06)
    ap.add_argument("--y_max", type=float, default=0.06)
    ap.add_argument("--feasible_mask", default=str(WORKTREE_ROOT / "datasets/feasible_mask_v3.json"))
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    state_dim = STATE_DIMS[args.state_history]
    action_dim = 4 * args.n_act
    print(f"Loading {args.checkpoint_dir} (state_dim={state_dim} action_dim={action_dim} "
          f"action_mode={args.action_mode})")
    policy = load_policy(args.checkpoint_dir, state_dim, action_dim, args.device)

    B = args.n_episodes
    client = ServerClient(args, B)
    reports = [None] * B
    try:
        imgs, aps = client.reset_all()
        recorded = np.zeros(B, dtype=bool)
        returns = np.zeros(B, dtype=np.float64)
        for step in range(args.max_steps // args.n_act + 1):
            if recorded.all():
                break
            obs = wrap_obs(imgs, aps, args.state_history, args.device)
            with torch.no_grad():
                chunk_actions, _ = policy.predict_action_batch(
                    obs, mode="eval", calculate_logprobs=False, calculate_values=False,
                    return_obs=False)
            actions_np = chunk_actions.detach().cpu().numpy().astype(np.float32).reshape(B, args.n_act, 4)
            imgs, aps, _fi, _fa, rewards, terms, truncs, succ, done_info = client.step_chunk(actions_np)
            returns[~recorded] += rewards[~recorded]
            for i in np.flatnonzero((terms | truncs) & ~recorded):
                reports[i] = done_info[i]
            recorded |= terms | truncs
            if (step + 1) % 20 == 0:
                print(f"  chunk {step+1}: {recorded.sum()}/{B} episodes recorded")
    finally:
        client.close()

    done = [r for r in reports if r is not None]
    modes = [classify(r) for r in done]
    counts = {k: modes.count(k) for k in ["success", "timeout", "overshoot", "undershoot", "tipped"]}
    sr = counts["success"] / B
    print(f"\n{'='*60}\ncheckpoint: {args.checkpoint_dir}")
    print(f"state_history={args.state_history} action_mode={args.action_mode} "
          f"N={B} seed={args.eval_seed_base} recorded={len(done)}")
    print(f"SUCCESS RATE: {100*sr:.1f}% ({counts['success']}/{B})")
    print("failure modes:", {k: v for k, v in counts.items() if k != "success"})
    print(f"mean return: {returns.mean():.3f}  "
          f"mean steps: {np.mean([r['steps'] for r in done]):.0f}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(dict(
            checkpoint=args.checkpoint_dir, state_history=args.state_history,
            action_mode=args.action_mode, max_step=args.max_step, n_episodes=B,
            seed=args.eval_seed_base, success_rate=sr, counts=counts,
            mean_return=float(returns.mean()), episodes=reports), indent=1))
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()

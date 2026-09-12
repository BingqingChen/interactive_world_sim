"""Closed-loop success-rate evaluation of a trained RLPD-in-real-sim checkpoint,
matching this project's standard evaluation protocol: N deterministic episodes,
success = AlohaChunkEnv's own eval_success gate (angle in [-105,-80] deg AND
flatness), same criterion eval_dp_rotate_t.py/AlohaChunkEnv already use for the
DP/PPO policies in this project.

Loads the CNNPolicy checkpoint IN THIS PROCESS (RLinf's venv, where torch +
the pretrained resnet encoder live) and bridges every env step to
real_sim_chunk_server.py (run under IWS's own venv) -- the same
persistent-subprocess architecture IWSRotateTSimEnv uses during training, for
the same reason (dm_control(EGL) + SAPIEN's IK solver segfault together in
RLinf's venv). Deterministic action = mode="eval" in CNNPolicy._generate_actions
(the tanh-squashed action MEAN, not a sampled action) -- the actual learned
policy's best-effort behavior, not exploration noise.

Runs num_envs = n_episodes in parallel (one real episode per row); each row's
FIRST completion (done=True) is recorded as that episode's result, and
further steps for an already-recorded row are computed but ignored (the
server auto-resets that row internally regardless).

Usage:
  MUJOCO_GL=egl /home/jacobhb/RLinf/.venv/bin/python \
      rlinf_integration/evaluate_rlpd_checkpoint.py \
      --checkpoint_dir /home/jacobhb/RLinf/results/.../checkpoints/global_step_1800/actor \
      --state_history single --n_episodes 20 --eval_seed_base 9000 --device cuda:0
"""
import argparse
import json
import os
import pickle
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

WORKTREE_ROOT = Path(__file__).resolve().parent.parent
IWS_VENV_PYTHON = "/home/jacobhb/projects/worth_doing/interactive_world_sim/.venv/bin/python"

sys.path.insert(0, str(WORKTREE_ROOT / "rlinf_integration"))

from rlinf.models.embodiment.cnn_policy.cnn_policy import CNNPolicy, CNNConfig  # noqa: E402

RESNET_MODEL_PATH = "/home/jacobhb/projects/intern_project/RLinf/RLinf-ResNet10-pretrained"


def load_policy(checkpoint_dir, state_dim, action_dim, device):
    cfg = CNNConfig()
    cfg.update_from_dict(dict(
        image_size=[3, 128, 128], image_num=1, action_dim=action_dim, state_dim=state_dim,
        num_action_chunks=1, backbone="resnet", model_path=RESNET_MODEL_PATH,
        encoder_config={"ckpt_name": "resnet10_pretrained.pt"},
        add_value_head=False, add_q_head=True, num_q_heads=10,
    ))
    model = CNNPolicy(cfg)
    weights_path = Path(checkpoint_dir) / "model_state_dict" / "full_weights.pt"
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: missing={missing} unexpected={unexpected}")
    model = model.to(device).eval()
    return model


class ServerClient:
    """Same binary IPC protocol as IWSRotateTSimEnv/real_sim_chunk_server.py."""

    def __init__(self, num_envs, max_steps, n_obs, x_range, y_range, feasible_mask, env_seed_base):
        server_script = WORKTREE_ROOT / "rlinf_integration" / "real_sim_chunk_server.py"
        self.proc = subprocess.Popen(
            [IWS_VENV_PYTHON, "-u", str(server_script),
             "--num_envs", str(num_envs), "--max_steps", str(max_steps),
             "--n_obs", str(n_obs),
             "--x_min", str(x_range[0]), "--x_max", str(x_range[1]),
             "--y_min", str(y_range[0]), "--y_max", str(y_range[1]),
             "--feasible_mask", str(feasible_mask),
             "--env_seed_base", str(env_seed_base),
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
    """imgs (B,n_obs,3,128,128) f32 [0,1] CHW, aps (B,n_obs,4) f32 -- matches
    IWSRotateTSimEnv._wrap_obs exactly."""
    last_img = imgs[:, -1]
    main_images = torch.from_numpy(last_img).to(device)
    main_images = (main_images.permute(0, 2, 3, 1) * 255.0)
    if state_history == "single":
        states_np = aps[:, -1]
    else:
        states_np = np.concatenate([aps[:, -2], aps[:, -1]], axis=-1)
    states = torch.from_numpy(states_np).to(device)
    return {"main_images": main_images, "states": states}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--state_history", choices=["single", "concat_state"], required=True)
    ap.add_argument("--n_episodes", type=int, default=20)
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
    args = ap.parse_args()

    state_dim = 4 if args.state_history == "single" else 8
    action_dim = 4 * args.n_act
    device = args.device

    print(f"Loading policy from {args.checkpoint_dir} (state_dim={state_dim} action_dim={action_dim})")
    policy = load_policy(args.checkpoint_dir, state_dim, action_dim, device)

    B = args.n_episodes
    client = ServerClient(
        num_envs=B, max_steps=args.max_steps, n_obs=args.n_obs,
        x_range=(args.x_min, args.x_max), y_range=(args.y_min, args.y_max),
        feasible_mask=args.feasible_mask, env_seed_base=args.eval_seed_base,
    )
    try:
        imgs, aps = client.reset_all()
        recorded = np.zeros(B, dtype=bool)
        successes = np.zeros(B, dtype=bool)
        rewards_sum = np.zeros(B, dtype=np.float64)
        n_chunks = np.zeros(B, dtype=np.int64)

        max_chunks = args.max_steps // args.n_act + 1
        for step in range(max_chunks):
            if recorded.all():
                break
            obs = wrap_obs(imgs, aps, args.state_history, device)
            with torch.no_grad():
                chunk_actions, _ = policy.predict_action_batch(obs, mode="eval", calculate_logprobs=False, calculate_values=False, return_obs=False)
            actions_np = chunk_actions.detach().cpu().numpy().astype(np.float32)  # (B,1,32)
            actions_np = actions_np.reshape(B, args.n_act, 4)
            imgs, aps, _final_imgs, _final_aps, rewards, terminations, truncations, succ = client.step_chunk(actions_np)
            done = terminations | truncations
            newly_done = done & ~recorded
            successes[newly_done] = succ[newly_done]
            rewards_sum[~recorded] += rewards[~recorded]
            n_chunks[~recorded] += 1
            recorded |= done
            if (step + 1) % 10 == 0:
                print(f"  step {step+1}: {recorded.sum()}/{B} episodes recorded")

        print(f"\n{'='*60}")
        print(f"checkpoint: {args.checkpoint_dir}")
        print(f"state_history: {args.state_history}  n_episodes: {B}  eval_seed_base: {args.eval_seed_base}")
        print(f"episodes recorded: {recorded.sum()}/{B}")
        print(f"SUCCESS RATE: {successes.mean()*100:.1f}%  ({successes.sum()}/{B})")
        print(f"mean chunks to done: {n_chunks.mean():.1f}")
        print(f"mean episode return: {rewards_sum.mean():.3f}")
        print("per-episode success:", successes.astype(int).tolist())
    finally:
        client.close()


if __name__ == "__main__":
    main()

"""IWSRotateTSimEnv -- validates the RLPD recipe against the REAL simulator (ground
truth angle+flatness reward, no WM proxy-estimator uncertainty at all), run in
parallel with tuning the WM's proxy reward estimator. Per the "remind me the reward
function" discussion: reuses AlohaChunkEnv verbatim (from ppo_residual_rotate_t.py,
the already-proven real-sim PPO precedent) rather than re-deriving the reward, so
this gets the EXACT formula that precedent validated -- full eval_success gate
(angle range [-105,-80] deg AND flatness), not the WM env's simplified angle-only
approximation.

Architecture: unlike IWSRotateTWorldEnv (where only the real reset is bridged to a
subprocess, since the WM itself needs to run in-process for GPU access), THIS env
bridges EVERY chunk step to a persistent subprocess (real_sim_chunk_server.py, run
under IWS's own venv) -- the entire environment (physics, action processing, reward)
lives there. This process (RLinf's venv) never imports dm_control/SAPIEN/mujoco at
all for this env, avoiding the segfault entirely rather than working around it, and
is a much simpler class as a result: it is a pure IPC client, no torch-heavy WM
loading, no angle-estimation logic.

Reset semantics also differ from IWSRotateTWorldEnv: real resets are cheap (no WM
GPU inference involved), so real_sim_chunk_server.py does true PER-ROW independent
auto-reset (matching AlohaChunkEnv's own usage convention in
ppo_residual_rotate_t.py), not the WM env's batch-synchronous reset (which exists
specifically because regenerating one row's WM video independently mid-batch is
awkward/expensive -- not a constraint here).
"""

import os
import pickle
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from rlinf.envs.world_model.base_world_env import BaseWorldEnv

IWS_ROOT = Path("/home/jacobhb/projects/worth_doing/interactive_world_sim/.claude/worktrees/rlpd-imagined-rotate-t")
IWS_VENV_PYTHON = "/home/jacobhb/projects/worth_doing/interactive_world_sim/.venv/bin/python"


class IWSRotateTSimEnv(BaseWorldEnv):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics=True):
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics)

    # ------------------------------------------------------------------ setup
    def _build_dataset(self, cfg):
        """Spawn the persistent real-sim server (see module docstring for why this
        env is a pure IPC client with no simulator dependency in-process). Returns
        None -- there is no fixed "dataset"; each real reset generates a fresh state
        live, same as IWSRotateTWorldEnv."""
        self.n_act = int(cfg.get("n_action_steps", 8))
        max_steps = int(cfg.get("max_steps", 800))  # RAW env steps (AlohaChunkEnv's own units, not chunks)
        n_obs = int(cfg.get("n_obs_steps", 2))
        x_range = tuple(cfg.get("x_range", (-0.06, 0.06)))
        y_range = tuple(cfg.get("y_range", (-0.06, 0.06)))
        feasible_mask_path = cfg.get(
            "feasible_mask", str(IWS_ROOT / "datasets/feasible_mask_v3.json")
        )
        env_seed_base = int(cfg.get("env_seed_base", 42))

        server_script = Path(__file__).parent / "real_sim_chunk_server.py"
        self._sim_proc = subprocess.Popen(
            [IWS_VENV_PYTHON, "-u", str(server_script),
             "--num_envs", str(self.num_envs), "--max_steps", str(max_steps),
             "--n_obs", str(n_obs),
             "--x_min", str(x_range[0]), "--x_max", str(x_range[1]),
             "--y_min", str(y_range[0]), "--y_max", str(y_range[1]),
             "--feasible_mask", str(feasible_mask_path),
             "--env_seed_base", str(env_seed_base),
             "--iws_scripts_root", str(IWS_ROOT)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            env={**os.environ, "MUJOCO_GL": "egl"},
        )
        return None

    def close(self):
        """Terminate the real-sim server subprocess. See IWSRotateTWorldEnv.close --
        same caveat: not called automatically by RLinf, call explicitly."""
        proc = getattr(self, "_sim_proc", None)
        if proc is not None and proc.poll() is None:
            proc.stdin.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()

    # ------------------------------------------------------------------ IPC
    def _send(self, req_byte, payload=None):
        proc = self._sim_proc
        if proc.poll() is not None:
            raise RuntimeError(
                f"real_sim_chunk_server died (exit code {proc.returncode})"
            )
        proc.stdin.write(req_byte)
        if payload is not None:
            data = pickle.dumps(payload, protocol=4)
            proc.stdin.write(struct.pack(">Q", len(data)))
            proc.stdin.write(data)
        proc.stdin.flush()
        header = proc.stdout.read(8)
        if len(header) < 8:
            raise RuntimeError("real_sim_chunk_server closed its stdout unexpectedly")
        (length,) = struct.unpack(">Q", header)
        return pickle.loads(proc.stdout.read(length))

    def _wrap_obs(self, imgs, aps):
        """Matches this project's own obs-dict convention (same as IWSRotateTWorldEnv
        and diffusion_unet_hybrid_image_policy.predict_action): {"image":
        (B,n_obs,3,128,128) f32 [0,1], "agent_pos": (B,n_obs,4) f32}. imgs/aps come
        directly from AlohaChunkEnv's own np.stack(self.img_h)/np.stack(self.ap_h),
        no reshaping needed."""
        image = torch.from_numpy(imgs).to(self.device)
        agent_pos = torch.from_numpy(aps).to(self.device)
        return {"image": image, "agent_pos": agent_pos}

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options: Optional[dict] = None):
        imgs, aps = self._send(b"\x01")
        self._reset_metrics()
        self._is_start = False
        return self._wrap_obs(imgs, aps), {}

    # -------------------------------------------------------------- chunk_step
    def chunk_step(self, actions):
        """actions: (B, n_act, 4) raw EE-xy targets. Returns the same 5-tuple
        contract as IWSRotateTWorldEnv.chunk_step (verified against env_worker.py's
        actual consumption, not the BaseWorldEnv abstract docstring -- see that
        env's docstring for the discrepancy)."""
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy().astype(np.float32)
        else:
            actions_np = np.asarray(actions, dtype=np.float32)

        imgs, aps, rewards, terminations, truncations, successes = self._send(b"\x02", actions_np)
        obs = self._wrap_obs(imgs, aps)
        device = self.device
        rewards_t = torch.from_numpy(rewards).to(device)
        term_t = torch.from_numpy(terminations).to(device)
        trunc_t = torch.from_numpy(truncations).to(device)

        B, n_act = self.num_envs, self.n_act
        chunk_rewards = torch.zeros(B, n_act, dtype=torch.float32, device=device)
        chunk_rewards[:, -1] = rewards_t
        chunk_terminations = torch.zeros(B, n_act, dtype=torch.bool, device=device)
        chunk_terminations[:, -1] = term_t
        chunk_truncations = torch.zeros(B, n_act, dtype=torch.bool, device=device)
        chunk_truncations[:, -1] = trunc_t

        infos = self._record_metrics(rewards_t, term_t, {})
        infos["success"] = successes

        return ([obs], chunk_rewards, chunk_terminations, chunk_truncations, [infos])

    # -------------------------------------------------------------- step
    def step(self, actions):
        """Not used by the actual RLPD/SAC rollout path (env_worker.py calls
        chunk_step exclusively -- see IWSRotateTWorldEnv's verification of this).
        Kept as a thin delegate for parity."""
        actions_t = torch.as_tensor(actions, dtype=torch.float32)
        if actions_t.dim() == 2:
            actions_t = actions_t.unsqueeze(1)
        obs_list, rewards, terms, truncs, infos_list = self.chunk_step(actions_t)
        return obs_list[-1], rewards[:, -1], terms[:, -1], truncs[:, -1], infos_list[-1]

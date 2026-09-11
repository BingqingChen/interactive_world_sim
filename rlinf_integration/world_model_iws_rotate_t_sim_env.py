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
        # "single": main_images/states are RLinf's native format exactly (current
        # frame only, matching WanEnv._wrap_obs and CNNPolicy.get_dummy_input's
        # (B,H,W,C) image / (B,state_dim) state contract -- no frame-history dim at
        # all). "concat_state": same single-frame image, but states is the
        # concatenation of the previous+current EE-xy (state_dim doubles) to keep
        # velocity-like information without touching the image channel count (which
        # would risk breaking the pretrained ResNet encoder's first-conv-layer shape).
        # Run both side-by-side (2 GPUs available) rather than guess which matters
        # for this task -- see the plan doc for the reasoning.
        self.state_history = cfg.get("state_history", "single")
        assert self.state_history in ("single", "concat_state")

        # .resolve() is required: this module is loaded via a symlink into
        # ~/RLinf/rlinf/envs/world_model/, and __file__ reports the symlink's own
        # path -- Path(__file__).parent without resolving would point at the RLinf
        # dir (where real_sim_chunk_server.py doesn't exist), not this worktree.
        server_script = Path(__file__).resolve().parent / "real_sim_chunk_server.py"
        self._sim_proc = subprocess.Popen(
            [IWS_VENV_PYTHON, "-u", str(server_script),
             "--num_envs", str(self.num_envs), "--max_steps", str(max_steps),
             "--n_obs", str(n_obs), "--action_repeat", str(self.n_act),
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
        """RLinf's OWN obs-dict contract (CNNPolicy.preprocess_env_obs /
        get_dummy_input, matching WanEnv._wrap_obs) -- NOT this project's
        diffusion_policy convention ({"image":(B,n_obs,3,H,W) f32 [0,1] CHW,
        "agent_pos":...}), which an earlier version of this env (and
        IWSRotateTWorldEnv, not yet fixed there) mistakenly used out of habit and
        would have crashed on the first real forward pass (KeyError: 'main_images').
        RLinf expects:
          "main_images": (B,H,W,3) HWC, values in [0,255] (CNNPolicy does /255.0
              internally in preprocess_env_obs -- do NOT pre-normalize here)
          "states": (B,state_dim) -- a single flat vector, no frame-history dim
        imgs/aps arrive as (B,n_obs=2,...) from AlohaChunkEnv's own 2-deep obs
        history; this env only ever uses the LATEST (most recent) frame for
        main_images (single-frame, per RLinf's native convention -- concatenating
        frames channel-wise would change the pretrained ResNet encoder's expected
        input channel count, a real risk not worth taking here). For states,
        self.state_history controls whether the previous frame's state is folded in
        too (state_dim doubles to preserve velocity-like info) -- see
        _build_dataset's comment for why this is being run as a side-by-side
        comparison rather than decided in advance."""
        last_img = imgs[:, -1]  # (B,3,128,128) f32 [0,1] CHW
        main_images = torch.from_numpy(last_img).to(self.device)
        main_images = (main_images.permute(0, 2, 3, 1) * 255.0)  # -> (B,128,128,3) [0,255]

        if self.state_history == "single":
            states_np = aps[:, -1]  # (B,4)
        else:  # concat_state
            states_np = np.concatenate([aps[:, -2], aps[:, -1]], axis=-1)  # (B,8)
        states = torch.from_numpy(states_np).to(self.device)

        return {"main_images": main_images, "states": states}

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options: Optional[dict] = None):
        imgs, aps = self._send(b"\x01")
        self._reset_metrics()
        self._is_start = False
        return self._wrap_obs(imgs, aps), {}

    # -------------------------------------------------------------- chunk_step
    def chunk_step(self, actions):
        """actions: (B, 1, 4) raw EE-xy target -- ONE action per call, not an
        8-long DP-style chunk. Corrected after a real smoke-test crash:
        CNNPolicy._generate_actions reshapes its action head's raw output to
        (-1, num_action_chunks, action_dim) with NO widening for
        num_action_chunks>1 (the head's output size is fixed at action_dim
        regardless), so num_action_chunks must be 1 for this policy family --
        confirmed by the crash ("shape '[-1, 8, 4]' is invalid for input of size
        16" when num_action_chunks was set to 8). RLinf therefore calls
        chunk_step once per SINGLE raw action, not once per 8-step DP chunk as
        originally designed.

        To preserve the validated reward granularity anyway (AlohaChunkEnv's
        dense progress reward is computed once per 8 raw physics sub-steps,
        matching this project's n_action_steps=8 DP convention), this env uses
        **action repeat**: the actor's one EE-xy target is held constant and
        fed to AlohaChunkEnv.step_chunk as an 8-long repeated chunk, entirely
        inside real_sim_chunk_server.py (see its --action_repeat arg) -- a
        standard RL temporal-abstraction pattern, invisible to RLinf as a
        multi-width tensor. Returns the same 5-tuple contract as
        IWSRotateTWorldEnv.chunk_step, but now with width 1 (=
        actor.model.num_action_chunks) along the chunk axis, not width n_act."""
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy().astype(np.float32)
        else:
            actions_np = np.asarray(actions, dtype=np.float32)
        actions_np = actions_np[:, 0]  # (B, 1, 4) -> (B, 4): the single action this call carries

        imgs, aps, rewards, terminations, truncations, successes = self._send(b"\x02", actions_np)
        obs = self._wrap_obs(imgs, aps)
        device = self.device
        rewards_t = torch.from_numpy(rewards).to(device)
        term_t = torch.from_numpy(terminations).to(device)
        trunc_t = torch.from_numpy(truncations).to(device)

        # Width 1 along the chunk axis, matching actor.model.num_action_chunks=1
        # (see chunk_step's docstring) -- NOT self.n_act (the internal 8-step
        # action-repeat count, which never leaves real_sim_chunk_server.py).
        chunk_rewards = rewards_t.unsqueeze(-1)
        chunk_terminations = term_t.unsqueeze(-1)
        chunk_truncations = trunc_t.unsqueeze(-1)

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

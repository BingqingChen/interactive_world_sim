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
        # Always a single current image. "none": image-only visuomotor policy,
        # states is (B,0) (needs CNNPolicy's state_dim=0 support). "single" /
        # "concat_state": latest / previous+latest EE-xy (showed no difference:
        # 46% vs 44% at N=50).
        self.state_history = cfg.get("state_history", "single")
        assert self.state_history in ("none", "single", "concat_state")
        # See real_sim_chunk_server.py: "delta" = normalized per-step deltas
        # integrated server-side; "absolute" = raw EE-xy targets.
        self.action_mode = cfg.get("action_mode", "absolute")
        assert self.action_mode in ("absolute", "delta")
        self.max_step = float(cfg.get("max_step", 0.04))

        # .resolve() is required: this module is loaded via a symlink into
        # ~/RLinf/rlinf/envs/world_model/, and __file__ reports the symlink's own
        # path -- Path(__file__).parent without resolving would point at the RLinf
        # dir (where real_sim_chunk_server.py doesn't exist), not this worktree.
        server_script = Path(__file__).resolve().parent / "real_sim_chunk_server.py"
        self._sim_proc = subprocess.Popen(
            [IWS_VENV_PYTHON, "-u", str(server_script),
             "--num_envs", str(self.num_envs), "--max_steps", str(max_steps),
             "--n_obs", str(n_obs),
             "--x_min", str(x_range[0]), "--x_max", str(x_range[1]),
             "--y_min", str(y_range[0]), "--y_max", str(y_range[1]),
             "--feasible_mask", str(feasible_mask_path),
             "--env_seed_base", str(env_seed_base),
             "--action_mode", self.action_mode, "--max_step", str(self.max_step),
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
        elif self.state_history == "concat_state":
            states_np = np.concatenate([aps[:, -2], aps[:, -1]], axis=-1)  # (B,8)
        else:
            # Image-only: a constant zero state (state_dim=1). CNNPolicy always has
            # a state branch; a constant input makes it a fixed learned bias that
            # carries no robot information, without patching RLinf.
            states_np = np.zeros((aps.shape[0], 1), np.float32)
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
        """actions: (B, 1, 32) -- CNNPolicy emits exactly ONE vector per forward
        call regardless of num_action_chunks (confirmed by a real crash trying
        num_action_chunks=8: its action head is Linear(hidden_dim, action_dim)
        with no widening). Rather than repeating a single action (which the
        WM was never trained on -- imagine_batch's dynamics_forward always
        consumes 8 genuinely distinct waypoints per call, so action-repeat is
        actually out-of-distribution), this env instead sets
        actor.model.action_dim = 4 * n_action_steps = 32 in the config, so the
        SAME unmodified CNNPolicy head naturally emits 8 waypoints' worth of
        values in one flat vector -- confirmed clean by reading the model code:
        the action head, Q-network, actor_logstd, and action_scale/bias are all
        generic in self.cfg.action_dim (cnn_policy.py:132,149,156,159,163-166),
        and action_scale is a single (lo,hi) pair broadcast uniformly across
        all 32 dims -- correct here since all 32 are the same physical
        quantity (EE-xy) repeated across the chunk, not heterogeneous joints.
        This env reshapes that flat (B,32) into a genuine (B,8,4) waypoint
        chunk and sends the WHOLE chunk to AlohaChunkEnv.step_chunk, exactly
        matching imagine_batch's own DP-chunk convention and the reward
        AlohaChunkEnv was actually validated against -- no action-repeat, no
        RLinf model-code changes.

        Returns the same 5-tuple contract as IWSRotateTWorldEnv.chunk_step,
        width 1 along the chunk axis (= actor.model.num_action_chunks, which
        stays 1 -- only action_dim widened, not num_action_chunks)."""
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy().astype(np.float32)
        else:
            actions_np = np.asarray(actions, dtype=np.float32)
        B = actions_np.shape[0]
        actions_np = actions_np.reshape(B, self.n_act, 4)  # (B,1,32) -> (B,8,4) genuine waypoint chunk

        (imgs, aps, final_imgs, final_aps, rewards, terminations, truncations, successes,
         done_info) = self._send(b"\x02", actions_np)
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
        # BaseWorldEnv._record_metrics already tracks self.success_once (sticky
        # True once terminations fires, reset per-episode by _reset_metrics) but
        # never surfaces it into infos["episode"] -- only "return" was being
        # logged, so eval/return was the only in-training signal and success
        # rate had to be eyeballed from its magnitude. env_worker.py pulls every
        # key out of infos["episode"] (masked by done) into env_info, which
        # compute_evaluate_metrics then .mean()s across envs and logs as
        # eval/<key> exactly like eval/return -- so this one line is enough to
        # get a real eval/success (a 0/1 mean = success RATE directly), no
        # RLinf-side (external repo) code touched.
        infos["episode"]["success"] = self.success_once.clone().float()
        # Failure modes of rows that finished this call (env_worker keeps only the
        # done rows' values), logged as eval/timeout etc. -- same classification
        # as evaluate_rlpd_checkpoint.classify.
        modes = {k: torch.zeros(self.num_envs, dtype=torch.float32, device=device)
                 for k in ("timeout", "overshoot", "undershoot", "tipped")}
        for i, r in enumerate(done_info):
            if r is None or r["success"]:
                continue
            if not (r["init_flat"] and r["final_flat"]):
                modes["tipped"][i] = 1.0
            elif r["delta_deg"] < -105.0:
                modes["overshoot"][i] = 1.0
            elif r["timeout"]:
                modes["timeout"][i] = 1.0
            else:
                modes["undershoot"][i] = 1.0
        infos["episode"].update(modes)

        done_t = term_t | trunc_t
        if done_t.any():
            # env_worker.py's env_interact_step unconditionally indexes
            # infos["final_info"]["..."] once ANY row is done (not guarded by an
            # "in infos" check on that particular line, unlike a neighboring one)
            # -- omitting this crashed the FIRST time any episode actually
            # completed (KeyError: 'final_info'), not caught by earlier short
            # smoke tests where episodes hadn't finished yet.
            #
            # "final_observation" is ALSO required, not just nice-to-have:
            # env_worker.py's append_transitions asserts next_obs is not None,
            # and next_obs = infos["final_observation"] whenever any row is done
            # -- confirmed via a second real crash (AssertionError in
            # append_transitions) after fixing the first. Unlike
            # IWSRotateTWorldEnv (batch-sync reset), real_sim_chunk_server.py
            # does per-row auto-reset transparently server-side (see module
            # docstring), so the pre-reset terminal obs isn't naturally
            # available here -- fixed by having the server report BOTH the
            # post-reset obs (imgs/aps, used for the NEXT step) AND each row's
            # own pre-reset terminal obs (final_imgs/final_aps) in every
            # step_chunk response, see that module's docstring.
            infos["final_observation"] = self._wrap_obs(final_imgs, final_aps)
            # dict(infos), NOT infos itself: `infos["final_info"] = infos` would
            # make infos contain itself (infos["final_info"]["final_info"]...
            # forever) -- confirmed via a real crash, RecursionError in
            # put_tensor_device's unguarded recursive nested-dict walk
            # (embodied_types.py's EnvOutput.__post_init__), the first time an
            # episode actually completed under a longer smoke run.
            infos["final_info"] = dict(infos)
            infos["_final_info"] = done_t
            infos["_final_observation"] = done_t

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

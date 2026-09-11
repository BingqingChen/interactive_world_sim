"""IWSRotateTWorldEnv -- wraps IWS's LatentWorldModel as an RLinf BaseWorldEnv, so
RLinf's existing RLPD implementation can be driven against it directly.

Chunk-granularity contract (one RL "step" = one 8-frame action chunk) mirrors
AlohaChunkEnv (interactive_world_sim/scripts/ppo_residual_rotate_t.py) and the WM
stepping body of imagine_batch (interactive_world_sim/scripts/collect_imagined_rotate_t.py)
exactly -- see the plan doc for the full design rationale.

Init distribution: random T XY within +-0.06m, restricted to feasible_mask_v3 --
matches the authoritative BC protocol (run_scaling_v3_sweep.py / ppo_residual_rotate_t.py
/ run_expertv3_pipeline.py), NOT +-0.08 (a stale default baked into older scripts).

Only frame 0 of each episode ever touches real physics (one AlohaEnv reset); every
frame after that is pure WM rollout, no resync -- driving the WM open-loop keeps it a
stationary transition function, which plain SAC/RLPD assumes.

Batch-reset convention: matches RLinf's own WanEnv (rlinf/envs/world_model/world_model_wan_env.py
_handle_auto_reset) rather than true per-row async reset -- when ANY row in the batch
finishes (success or step-budget), the WHOLE BATCH resets together. This is RLinf's own
precedent for world-model envs (regenerating one row's WM video sequence independently
mid-batch is awkward), not a shortcut invented here.

Reward mirrors AlohaChunkEnv.step_chunk's formula exactly:
    prog = angle_prev - angle_end
    reward = prog*5.0 - 0.01 + (5.0 if angle_end <= radians(-80) else 0)
but angle_end is read with a hardened estimator instead of one raw est_angle() call,
because pure-imagined WM frames showed the estimator going wildly wrong when the WM
itself hallucinates (see docs/experiment_log_push-T.md and the reward_verification_ep*
videos generated during planning: WM-proxy estimates diverged from ground truth by
35-57 degrees on identical action sequences). The hardening:
  - confidence (best-match IoU) is now returned alongside the angle (est_angle_with_conf)
  - a mask-area sanity check against frame 0's own mask area catches occlusion/garbled
    frames the existing `mask.sum() < 30` check misses
  - the last 3 decoded frames of each chunk are read (not just the final one) and
    combined by confidence-weighted median
  - a chunk-boundary jump exceeding JUMP_REJECT_DEG is rejected (angle carried forward)
    -- grounded empirically at ~45-50 deg: real per-chunk angle change never exceeded
    34 deg in a 24-episode ground-truth sample (p99 24 deg), while WM-proxy per-frame
    deltas had std 13.6 deg (16x the real std), max 147 deg. RECALIBRATE against the
    full real-demo pool before trusting this threshold for a real run (see plan doc).
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from rlinf.envs.world_model.base_world_env import BaseWorldEnv

# Points at the rlpd-imagined-rotate-t worktree, NOT the main IWS checkout -- this
# branch's est_angle_with_conf addition to collect_imagined_rotate_t.py (and any
# other rlpd-specific IWS-side changes) only exist there.
IWS_ROOT = Path("/home/jacobhb/projects/worth_doing/interactive_world_sim/.claude/worktrees/rlpd-imagined-rotate-t")
sys.path.insert(0, str(IWS_ROOT / "scripts"))
sys.path.insert(0, str(IWS_ROOT / "scripts" / "data_collection"))

import eval_dp_rotate_t as E  # noqa: E402  (DT, K_P, K_V, ACC_LIM, VEL_LIM, get_obs)
import collect_rotate_t as C  # noqa: E402  (stabilize_t, settle_arms, t_angle, make_upright_pose_sampler, UPRIGHT_TOL)
import gym_aloha.env as gae  # noqa: E402
from gym_aloha.env import AlohaEnv  # noqa: E402
from yixuan_utilities.kinematics_helper import KinHelper  # noqa: E402
from interactive_world_sim.utils.pose_utils import PoseType, pose_convert  # noqa: E402
from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm  # noqa: E402
import eval_metric_correlation as M  # noqa: E402  (load_viz_cfg, load_model)
from collect_imagined_rotate_t import (  # noqa: E402
    make_templates,
    est_angle_with_conf,
    TERMINAL_DEG,
)

N_OBS = 2  # matches n_obs_steps=2 project-wide (diffusion_unet_hybrid_rotate_t.yaml)


class IWSRotateTWorldEnv(BaseWorldEnv):
    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info,
        record_metrics: bool = True,
    ):
        # BaseWorldEnv.__init__ calls self._build_dataset(cfg) and _init_metrics(),
        # so all one-time setup (WM load, real env, pose sampler) happens inside
        # _build_dataset below, called from super().__init__.
        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
        )

    # ------------------------------------------------------------------ setup
    def _build_dataset(self, cfg):
        """One-time setup: load the WM, build the real reset env + pose sampler +
        feasible-mask region check. Returns None -- unlike WanEnv (which draws from a
        fixed pre-recorded initial-frame dataset), this env generates each episode's
        initial frame live from a real MuJoCo reset, so there is no fixed "dataset" to
        hold; self.dataset is unused by this env beyond the abstract-method contract."""
        self.n_act = int(cfg.get("n_action_steps", 8))
        self.wm_max_chunks = int(cfg.get("wm_max_chunks", 40))
        self.jump_reject_deg = float(cfg.get("jump_reject_deg", 47.5))
        # Hard floor on est_angle_with_conf's best-match IoU -- grounded empirically at
        # 0.5 (real frames never scored below 0.78 in a 24-episode sample; ~6% of WM
        # frames scored below 0.5, i.e. genuinely deformed, not just noisy). See
        # _robust_angle_end for the full rationale.
        self.min_iou = float(cfg.get("min_iou", 0.5))
        self.terminal_deg = float(cfg.get("terminal_deg", TERMINAL_DEG))
        self.dec_infer_steps = int(cfg.get("dec_infer_steps", 2))
        x_range = tuple(cfg.get("x_range", (-0.06, 0.06)))
        y_range = tuple(cfg.get("y_range", (-0.06, 0.06)))
        feasible_mask_path = cfg.get(
            "feasible_mask", str(IWS_ROOT / "datasets/feasible_mask_v3.json")
        )
        wm_ckpt = cfg.get("wm_ckpt", "ckpts/push_t/epoch=3-step=90000.ckpt")
        wm_config = cfg.get("wm_config", "pusht_mujoco")
        env_seed_base = int(cfg.get("env_seed_base", 700_000))

        import json

        self.env_seed_base = env_seed_base
        self._reset_trial = 0

        vcfg = M.load_viz_cfg(wm_config)
        self.wm = M.load_model(wm_ckpt, vcfg.algorithm, self._get_runtime_device_str())
        self.wm.dec_infer_steps = self.dec_infer_steps

        self._real_env = AlohaEnv("pusht")
        self._kin = KinHelper(robot_name="trossen_vx300s")
        gae.sample_pusht_pose = C.make_upright_pose_sampler(x_range, y_range)

        m = json.loads(Path(feasible_mask_path).read_text())
        self._mask = np.array(m["mask"], bool)
        self._edges = np.array(m["grid_cm"]["edges"])

        return None

    def _in_feasible_region(self, env):
        s = env._env.task.get_observation(env._env.physics)["env_state"]
        x_cm, y_cm = s[0] * 100, s[1] * 100
        e = self._edges
        if not (e[0] <= x_cm < e[-1] and e[0] <= y_cm < e[-1]):
            return False
        i, j = int(np.digitize(x_cm, e) - 1), int(np.digitize(y_cm, e) - 1)
        return bool(self._mask[i, j])

    def _one_real_reset(self):
        """One real AlohaEnv reset -> (image f32 (3,128,128) [0,1], home_xy f32 (4,),
        frame_u8 (128,128,3)). Retries until upright + inside the feasible mask,
        mirroring get_initial_states' region-guard loop in collect_imagined_rotate_t.py."""
        env = self._real_env
        while True:
            np.random.seed(self.env_seed_base + self._reset_trial)
            env.reset(seed=self.env_seed_base + self._reset_trial)
            self._reset_trial += 1
            C.stabilize_t(env)
            o = env._env.task.get_observation(env._env.physics)
            lb = pose_convert(o["left_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            rb = pose_convert(o["right_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            world_t_bases = np.stack([lb, rb])
            C.settle_arms(
                env, world_t_bases, self._kin, np.zeros(6),
                E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM,
            )
            if abs(C.t_angle(env)) > C.UPRIGHT_TOL:
                continue
            if not self._in_feasible_region(env):
                continue
            image, agent_pos, frame_u8 = E.get_obs(env, self._kin, world_t_bases)
            return image, agent_pos.astype(np.float32), frame_u8

    # ------------------------------------------------------------------ reset
    @torch.no_grad()
    def reset(self, *, seed=None, options: Optional[dict] = None):
        B, device = self.num_envs, self.device
        imgs0, homes, frames0 = [], [], []
        for _ in range(B):
            img, home, frame = self._one_real_reset()
            imgs0.append(img)
            homes.append(home)
            frames0.append(frame)
        img0 = np.stack(imgs0)  # (B,3,128,128) f32 [0,1]
        home_xy = np.stack(homes).astype(np.float32)  # (B,4)
        frame0_u8 = np.stack(frames0)  # (B,128,128,3) u8

        f0 = torch.from_numpy(img0).to(device)
        f0n = self.wm.normalizer["top_pov"].normalize(f0)
        z0 = self.wm.encoder_forward(f0n).to(self.wm.dtype)  # (B,C,H,W)
        self.z_hist = z0.unsqueeze(1)  # (B,1,C,H,W)

        self.prev_img = f0.clone()
        self.curr_img = f0.clone()
        self.prev_state = torch.from_numpy(home_xy).to(device)
        self.curr_state = self.prev_state.clone()

        cap = self.wm_max_chunks * self.n_act + self.n_act
        self.exec_actions_hist = torch.zeros(B, cap, 4, dtype=torch.float32)
        self.t = 0
        self.steps = 0

        # per-row template library (each row's own upright frame0 anchors its estimator)
        self._templates, self._tc, self._frame0_area = [], [], []
        for b in range(B):
            tpl, tc = make_templates(frame0_u8[b])
            self._templates.append(tpl)
            self._tc.append(tc)
            from collect_imagined_rotate_t import red_mask
            self._frame0_area.append(int(red_mask(frame0_u8[b]).sum()))

        self.angle_prev = torch.zeros(B, dtype=torch.float32)  # upright init => angle 0
        self.frame0_u8 = frame0_u8
        self._jump_reject_count = getattr(self, "_jump_reject_count", 0)
        self._lost_track_count = getattr(self, "_lost_track_count", 0)  # no mask / area sanity fail on every candidate frame
        self._morph_reject_count = getattr(self, "_morph_reject_count", 0)  # mask present & right-sized, but IoU below floor on every candidate frame

        self._reset_metrics()
        self._is_start = False

        obs = self._wrap_obs()
        return obs, {}

    # ------------------------------------------------------------------ obs
    def _wrap_obs(self):
        """Matches this project's own obs-dict convention (diffusion_unet_hybrid_image_policy
        predict_action / eval_dp_rotate_t.get_obs): {"image": (B,2,3,128,128) f32 [0,1],
        "agent_pos": (B,2,4) f32}, last axis of the 2-length obs window = most recent."""
        image = torch.stack([self.prev_img, self.curr_img], dim=1)
        agent_pos = torch.stack([self.prev_state, self.curr_state], dim=1)
        return {"image": image, "agent_pos": agent_pos}

    # -------------------------------------------------------------- reward
    def _robust_angle_end(self, dec_u8_last3, row):
        """dec_u8_last3: list of up-to-3 uint8 (128,128,3) frames (most recent last),
        for row `row`. Returns (angle_end_rad, accepted: bool)."""
        templates, tc = self._templates[row], self._tc[row]
        f0_area = self._frame0_area[row]
        prev = float(self.angle_prev[row])

        reads = []  # (angle, iou) for frames that pass BOTH sanity checks
        any_mask_ok = False  # at least one frame had a plausible-area mask (area check passed)
        for frame in dec_u8_last3:
            angle, iou, area = est_angle_with_conf(frame, templates, tc)
            if angle is None:
                continue
            # area sanity: reject if the mask ballooned/collapsed vs frame 0's own area
            # (fixed overhead camera -> the T's projected area shouldn't swing wildly
            # under pure rotation; a big deviation flags occlusion/color hallucination)
            if area < 0.3 * f0_area or area > 3.0 * f0_area:
                continue
            any_mask_ok = True
            # HARD IoU floor -- catches "T morphed into some other shape" (roughly
            # T-sized so the area check alone misses it, but no rigid rotation of the
            # template explains it well). Grounded empirically: across a 24-episode
            # real-vs-imagined paired sample, REAL frames never scored below IoU=0.78
            # (min over 4800 frames) against their own best-fit rotation, while ~6% of
            # WM frames scored below 0.5 (worst 0.13) -- a clear separation. Below this
            # floor the read is dropped entirely, NOT just down-weighted: previously
            # min_iou only fed the confidence-weighted median as a soft weight, so a
            # single badly-deformed read with e.g. iou=0.15 still got normalized weight
            # 1.0 and was fully trusted when it was the only candidate in the window.
            if iou < self.min_iou:
                continue
            reads.append((angle, iou))

        if not reads:
            if any_mask_ok:
                self._morph_reject_count += 1  # had a plausible mask, but shape didn't match any rotation
            else:
                self._lost_track_count += 1  # no mask at all, or wildly wrong area
            return prev, False  # carry forward, not accepted (prog=0 for this chunk)

        # confidence-weighted median: sort by angle, pick the read whose cumulative
        # IoU-weight crosses the halfway point (a simple, dependency-free weighted
        # median -- good enough for up to 3 candidates).
        reads.sort(key=lambda r: r[0])
        weights = np.array([r[1] for r in reads], dtype=np.float64)
        weights = weights / max(weights.sum(), 1e-8)
        cum = np.cumsum(weights)
        idx = int(np.searchsorted(cum, 0.5))
        idx = min(idx, len(reads) - 1)
        candidate = reads[idx][0]

        jump_deg = abs(np.degrees(candidate - prev))
        if jump_deg > self.jump_reject_deg:
            self._jump_reject_count += 1
            return prev, False

        return candidate, True

    # -------------------------------------------------------------- chunk_step
    @torch.no_grad()
    def chunk_step(self, actions):
        """actions: (B, n_act, 4) raw EE-xy targets, already prepared by
        rlinf.envs.action_utils.prepare_actions (pass-through for this env type --
        see rlinf_integration/ACTION_UTILS_PATCH.md).

        Returns ([obs], rewards (B,n_act), terminations (B,n_act), truncations (B,n_act),
        [infos]) -- matches WanEnv.chunk_step's actual return contract exactly (the
        BaseWorldEnv abstract-method docstring's "(obs, reward, done, info)" is NOT
        what the real env_worker call site expects; verified against
        rlinf/workers/env/env_worker.py's chunk_step consumption)."""
        B, device, n_act = self.num_envs, self.device, self.n_act
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32)
        actions = actions.detach().to("cpu", dtype=torch.float32)  # (B, n_act, 4)

        t = self.t
        self.exec_actions_hist[:, t : t + n_act] = actions
        self.exec_actions_hist[:, t + n_act] = actions[:, -1]  # trailing pad

        lo = max(0, t - 9)
        act_win = self.exec_actions_hist[:, lo : t + n_act + 1].to(device)
        act_win = self.wm.normalizer["action"].normalize(act_win).to(self.wm.dtype)
        z_new = self.wm.dynamics_forward(self.z_hist, act_win)  # (B,n_act,C,H,W)
        self.z_hist = torch.cat([self.z_hist, z_new], dim=1)[:, -10:]

        Bn = B * n_act
        dec = render_img_cm(
            self.wm, z_new.reshape(Bn, *z_new.shape[2:]), resolution=128,
            normalizer=self.wm.normalizer, num_views=1, batch_size=16,
        ).float().clamp(0, 1).reshape(B, n_act, 3, 128, 128)
        dec_u8 = (dec * 255).round().byte().permute(0, 1, 3, 4, 2).cpu().numpy()  # (B,n_act,128,128,3)

        self.prev_img = dec[:, -2] if n_act >= 2 else self.curr_img
        self.curr_img = dec[:, -1]
        self.t += n_act
        t2 = self.t
        self.prev_state = self.exec_actions_hist[:, t2 - 2].to(device)
        self.curr_state = self.exec_actions_hist[:, t2 - 1].to(device)
        self.steps += 1

        k = min(3, n_act)
        rewards_last = torch.zeros(B, dtype=torch.float32)
        terminations_last = torch.zeros(B, dtype=torch.bool)
        for b in range(B):
            last_frames = [dec_u8[b, j] for j in range(n_act - k, n_act)]
            angle_end, _accepted = self._robust_angle_end(last_frames, b)
            prog = float(self.angle_prev[b]) - angle_end
            success = angle_end <= np.radians(self.terminal_deg)
            rewards_last[b] = prog * 5.0 - 0.01 + (5.0 if success else 0.0)
            terminations_last[b] = success
            self.angle_prev[b] = angle_end

        truncations_last = torch.tensor(
            [self.steps >= self.wm_max_chunks] * B, dtype=torch.bool
        )

        chunk_rewards = torch.zeros(B, n_act, dtype=torch.float32)
        chunk_rewards[:, -1] = rewards_last
        chunk_terminations = torch.zeros(B, n_act, dtype=torch.bool)
        chunk_terminations[:, -1] = terminations_last
        chunk_truncations = torch.zeros(B, n_act, dtype=torch.bool)
        chunk_truncations[:, -1] = truncations_last

        past_dones = terminations_last | truncations_last
        extracted_obs = self._wrap_obs()
        infos = {}
        if past_dones.any():
            # Batch-synchronous reset, matching RLinf's own WanEnv._handle_auto_reset
            # precedent for world-model envs (see module docstring).
            final_obs = extracted_obs
            infos["final_observation"] = final_obs
            infos["_final_observation"] = past_dones
            extracted_obs, _ = self.reset()

        infos = self._record_metrics(rewards_last, terminations_last, infos)
        infos["jump_reject_count"] = self._jump_reject_count
        infos["lost_track_count"] = self._lost_track_count
        infos["morph_reject_count"] = self._morph_reject_count

        return (
            [extracted_obs],
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            [infos],
        )

    # -------------------------------------------------------------- step
    def step(self, actions):
        """NOT used by the actual RLPD/SAC rollout path -- env_worker.py calls
        chunk_step exclusively (verified against rlinf/workers/env/env_worker.py).
        Kept as a thin delegate rather than a hard crash in case some other code
        path (e.g. a generic sanity check) calls it."""
        actions_t = torch.as_tensor(actions, dtype=torch.float32)
        if actions_t.dim() == 2:  # (B, action_dim) -> (B, 1, action_dim)
            actions_t = actions_t.unsqueeze(1)
        obs_list, rewards, terms, truncs, infos_list = self.chunk_step(actions_t)
        return obs_list[-1], rewards[:, -1], terms[:, -1], truncs[:, -1], infos_list[-1]

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

Per-row reset: when a row's episode ends (success, step budget or early stop), only that
row gets a new real reset; the other rows continue to their own normal end. Earlier
versions reset the WHOLE batch whenever any row finished (following RLinf's WanEnv
_handle_auto_reset), which cut every other row's episode short; changed 2026-09-13 per
user decision. Each row keeps its own step counter and WM context. dynamics_forward
depends only on the context length (a fresh episode's context grows 1 -> 9 -> 10 frames
over its first chunks), so rows are grouped by that length and a row reset mid-batch sees
exactly the context imagine_batch would give it.

Early stop (early_stop_rule="any_reject", user decision 2026-09-13): a chunk is bad if its
angle read is rejected (lost track / low-IoU shape / jump) or the T mask area leaves
[area_ratio_low, area_ratio_high] x frame 0 in any of the chunk's last 3 frames (growth =
morph_detected, shrink = shrink_detected). morph_terminate_patience (3) consecutive bad
chunks end the episode as a TRUNCATION, so the critic still bootstraps from the final
frame. early_stop_rule="morph_only" restores the earlier rule (area growth only).

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

# Points at the rlpd-imagined-rotate-t worktree, NOT the main IWS checkout -- this
# branch's est_angle_with_conf addition to collect_imagined_rotate_t.py (and any
# other rlpd-specific IWS-side changes) only exist there.
IWS_ROOT = Path("/home/jacobhb/projects/worth_doing/interactive_world_sim/.claude/worktrees/rlpd-imagined-rotate-t")
sys.path.insert(0, str(IWS_ROOT / "scripts"))
sys.path.insert(0, str(IWS_ROOT / "scripts" / "data_collection"))

# venvs aren't git-tracked so worktrees don't have their own -- always use the main
# checkout's, which is a stable, always-present resource independent of branch.
IWS_VENV_PYTHON = "/home/jacobhb/projects/worth_doing/interactive_world_sim/.venv/bin/python"

from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm  # noqa: E402
import eval_metric_correlation as M  # noqa: E402  (load_viz_cfg, load_model)
from collect_imagined_rotate_t import (  # noqa: E402
    make_templates,
    est_angle_with_conf,
    red_mask,
    TERMINAL_DEG,
)

N_OBS = 2  # matches n_obs_steps=2 project-wide (diffusion_unet_hybrid_rotate_t.yaml)

# Wider than collect_imagined_rotate_t.py's shared THETAS (-130..40 deg, tuned for
# DP-driven collection rollouts that stay near the task's target range). RL
# exploration routinely drives the T past that -- confirmed directly: a smooth_walk
# diagnostic run produced 6 spurious "jump" events, ALL landing at exactly -130.0 deg
# (5/6) or within 3.5 deg of it, i.e. the grid boundary, not a real WM discontinuity.
# Full 360 deg coverage so genuine continued rotation is always representable.
REWARD_THETAS = np.radians(np.arange(-180, 180, 1.5))


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
        """One-time setup: load the WM, spawn the persistent real-reset server.
        Returns None -- unlike WanEnv (which draws from a fixed pre-recorded
        initial-frame dataset), this env generates each episode's initial frame live
        from a real MuJoCo reset, so there is no fixed "dataset" to hold;
        self.dataset is unused by this env beyond the abstract-method contract.

        The real reset (AlohaEnv + KinHelper's SAPIEN-based IK) runs in a SEPARATE
        persistent subprocess under IWS's own venv, not in-process here -- running
        dm_control(EGL) and SAPIEN's IK solver together in RLinf's venv segfaults
        (a native-library conflict, root cause not found; reproduces even with
        package versions identical to IWS's own working venv -- see the plan doc's
        "Blocking issue" section for the full bisection). The server is spawned ONCE
        and answers reset requests for the rest of this env's life -- critical for
        real training, where every row's episode end triggers a real reset (see
        chunk_step), so a full run needs on the
        order of 1000+ real resets; a fresh subprocess per reset would pay full
        interpreter-plus-heavy-import startup cost every time, dwarfing the WM's own
        diffusion-sampling cost."""
        self.n_act = int(cfg.get("n_action_steps", 8))
        self.wm_max_chunks = int(cfg.get("wm_max_chunks", 40))
        self.jump_reject_deg = float(cfg.get("jump_reject_deg", 47.5))
        # Area-ratio (vs frame 0) sanity bounds. REVISED from an initial [0.3, 3.0] --
        # far too loose to catch WM hallucination in practice: a labeled dataset of
        # 800 chunk-frames across 20 smooth_walk episodes (2 genuinely morphed, via
        # direct visual inspection; 18 clean) showed [0.3,3.0] misses 97.8% of
        # genuinely morphed frames, because IoU's rotation search "explains away"
        # moderate deformation by finding SOME best-fit angle, while area growth is a
        # much more direct fingerprint of hallucinated pixels accumulating in the
        # mask (real physics keeps area within [0.86, 1.22] under a fixed overhead
        # camera; morphed frames in that same dataset ranged [1.10, 2.23], mostly well
        # above 1.25). [0.75, 1.20] gives 0.1% false-positive / 2.2% false-negative on
        # that labeled set (vs 0%/97.8% for the old bound) -- area_ratio_low is more
        # generous than area_ratio_high because occlusion (normal, expected -- the
        # gripper covers part of the T in every frame of this task) shrinks area,
        # while hallucination artifacts (color bleed, blur, extra red pixels) grow it;
        # these are different mechanisms and don't need a symmetric bound.
        self.area_ratio_low = float(cfg.get("area_ratio_low", 0.75))
        self.area_ratio_high = float(cfg.get("area_ratio_high", 1.20))
        # Hard floor on est_angle_with_conf's best-match IoU. REVISED from an initial
        # 0.5 down to 0.35 after direct evidence it was rejecting legitimate frames:
        # a smooth_walk diagnostic run's morph-rejects clustered at IoU 0.34-0.50
        # (mean 0.42, n=34) -- visually inspected several of the actual rejected
        # frames (see plan doc) and they show a correctly-shaped, correctly-rotated T
        # normally gripped by the bimanual arms, NOT a deformed blob. The gripper
        # mechanism overlaps part of the T's silhouette in essentially every frame of
        # this task (real or imagined), which lowers raw pixel-mask IoU against a
        # rigid unoccluded template regardless of whether the underlying shape is
        # correct -- the original 0.5 floor, calibrated only from aggregate real-vs-WM
        # IoU statistics (not from looking at what specific frames near the boundary
        # actually show), didn't account for this. 0.35 still clears the genuinely
        # degenerate tail from that same calibration (WM frames as low as IoU=0.13).
        self.min_iou = float(cfg.get("min_iou", 0.35))
        # After this many CONSECUTIVE rejected chunks for a row, force-accept the best
        # available area+IoU-passing candidate even if it fails the jump check (the
        # "stuck-prev recovery": a stale `prev` from one transient misread otherwise
        # makes every later read look "far from prev"; it was added for grid-boundary
        # misreads, which the full-circle REWARD_THETAS since fixed). 0 = disabled,
        # the default since 2026-09-13: under early_stop_rule="any_reject" the third
        # consecutive reject ends the episode, and force-accepting it would break the
        # run of bad chunks before patience is reached.
        self.stuck_reject_limit = int(cfg.get("stuck_reject_limit", 0))
        # Which chunks count toward morph_terminate_patience (see module docstring):
        # "any_reject" -- rejected read, area growth or area shrink (user decision
        # 2026-09-13); "morph_only" -- area growth only, the earlier rule. On the
        # 800-episode imagined pool at patience 3 (before shrink was counted;
        # outputs/reward_reject_patience_full_pool) morph_only cut 21 episodes (2.6%)
        # and any_reject 85 (10.6%); a visual sample showed the extra cuts are mostly
        # the T collapsing into an L/V shape (area shrinks, which growth-only morph
        # detection missed), plus some ambiguous jump streaks.
        self.early_stop_rule = cfg.get("early_stop_rule", "any_reject")
        assert self.early_stop_rule in ("any_reject", "morph_only"), self.early_stop_rule
        assert not (self.early_stop_rule == "any_reject" and self.stuck_reject_limit > 0), (
            "stuck-prev recovery would force-accept the reject that completes a bad run"
        )
        # Consecutive bad chunks (see early_stop_rule) required before truncating the
        # episode. Evidence for morph_detected alone: REVISED from immediate
        # (patience=1) truncation after validating
        # against the full 800-episode imagined pool (datasets/scaling_v3/imag/
        # {half1,half2}.zarr): 348/800 episodes (43.5%) had >=1 morph_detected chunk,
        # but only 21/800 (2.6%) ever reached a run of >=3 consecutive morph_detected
        # chunks -- 81.2% of morph_detected runs are length 1. Of isolated
        # (run-length<3) events with enough follow-up data, 98.8% (318/322) were
        # clean again within the next 3 chunks -- visually confirmed several of
        # these are transient one-frame WM rendering glitches (a dark smudge that
        # clears by the next frame), not sustained hallucination, versus a directly
        # confirmed genuine case (progressive mask growth over 13 consecutive
        # chunks, visually unambiguous blob growth) that patience=3 still catches
        # correctly, just 2 chunks later than immediate would have. Immediate
        # truncation was discarding ~43% of episodes for events that self-correct
        # ~99% of the time -- a real cost to online RL sample efficiency, not a
        # safety-only tradeoff, since the world only has finite parallel envs.
        self.morph_terminate_patience = int(cfg.get("morph_terminate_patience", 3))
        self.terminal_deg = float(cfg.get("terminal_deg", TERMINAL_DEG))
        self.dec_infer_steps = int(cfg.get("dec_infer_steps", 2))
        # Recipe parity with IWSRotateTSimEnv (sim runs A/B): "none" = image-only
        # policy (constant zero state); "delta" = normalized per-step deltas
        # integrated exactly as real_sim_chunk_server.py does, so the WM still
        # receives absolute EE-xy targets like the ones it was trained on.
        self.state_history = cfg.get("state_history", "single")
        assert self.state_history in ("none", "single")
        self.action_mode = cfg.get("action_mode", "absolute")
        assert self.action_mode in ("absolute", "delta")
        self.max_step = float(cfg.get("max_step", 0.04))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from real_sim_chunk_server import DEMO_TARGET_LO, DEMO_TARGET_HI
        self.target_lo = np.array(DEMO_TARGET_LO, np.float64)
        self.target_hi = np.array(DEMO_TARGET_HI, np.float64)
        x_range = tuple(cfg.get("x_range", (-0.06, 0.06)))
        y_range = tuple(cfg.get("y_range", (-0.06, 0.06)))
        feasible_mask_path = cfg.get(
            "feasible_mask", str(IWS_ROOT / "datasets/feasible_mask_v3.json")
        )
        wm_ckpt = cfg.get("wm_ckpt", "ckpts/push_t/epoch=3-step=90000.ckpt")
        wm_config = cfg.get("wm_config", "pusht_mujoco")
        env_seed_base = int(cfg.get("env_seed_base", 700_000))

        vcfg = M.load_viz_cfg(wm_config)
        self.wm = M.load_model(wm_ckpt, vcfg.algorithm, self._get_runtime_device_str())
        self.wm.dec_infer_steps = self.dec_infer_steps

        # .resolve() is required: this module is loaded via a symlink into
        # ~/RLinf/rlinf/envs/world_model/, and __file__ reports the symlink's own
        # path -- Path(__file__).parent without resolving would point at the RLinf
        # dir (where real_reset_server.py doesn't exist), not this worktree.
        server_script = Path(__file__).resolve().parent / "real_reset_server.py"
        # See IWSRotateTSimEnv: `egl_device_id` overrides RLinf's per-worker
        # MUJOCO_EGL_DEVICE_ID, which does not follow CUDA order on this box.
        server_env = {**os.environ, "MUJOCO_GL": "egl"}
        if cfg.get("egl_device_id", None) is not None:
            server_env["MUJOCO_EGL_DEVICE_ID"] = str(cfg.get("egl_device_id"))
        self._reset_proc = subprocess.Popen(
            [IWS_VENV_PYTHON, "-u", str(server_script),
             "--x_min", str(x_range[0]), "--x_max", str(x_range[1]),
             "--y_min", str(y_range[0]), "--y_max", str(y_range[1]),
             "--feasible_mask", str(feasible_mask_path),
             "--env_seed_base", str(env_seed_base),
             "--iws_scripts_root", str(IWS_ROOT)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            env=server_env,
        )
        return None

    def close(self):
        """Terminate the real-reset server subprocess. Not called automatically by
        RLinf (no standard env-teardown hook found) -- call explicitly when done, or
        rely on the OS cleaning it up when this process exits (best-effort only)."""
        proc = getattr(self, "_reset_proc", None)
        if proc is not None and proc.poll() is None:
            proc.stdin.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()

    def _one_real_reset(self):
        """One real AlohaEnv reset -> (image f32 (3,128,128) [0,1], home_xy f32 (4,),
        frame_u8 (128,128,3)), via a single request/response round-trip to the
        persistent real_reset_server subprocess (see _build_dataset's docstring for
        why this isn't done in-process)."""
        proc = self._reset_proc
        if proc.poll() is not None:
            raise RuntimeError(
                f"real_reset_server died (exit code {proc.returncode}) -- "
                "no restart logic yet, see plan doc's open items"
            )
        proc.stdin.write(b"\x01")
        proc.stdin.flush()
        header = proc.stdout.read(8)
        if len(header) < 8:
            raise RuntimeError("real_reset_server closed its stdout unexpectedly")
        (length,) = struct.unpack(">Q", header)
        payload = proc.stdout.read(length)
        image, home_xy, frame_u8 = pickle.loads(payload)
        return image, home_xy.astype(np.float32), frame_u8

    # ------------------------------------------------------------------ reset
    @torch.no_grad()
    def reset(self, *, seed=None, options: Optional[dict] = None):
        """Full reset: allocate per-row state, then real-reset every row."""
        B, device = self.num_envs, self.device
        cap = self.wm_max_chunks * self.n_act + self.n_act
        self.exec_actions_hist = torch.zeros(B, cap, 4, dtype=torch.float32)
        self.t = np.zeros(B, dtype=np.int64)  # raw WM steps into each row's episode
        self.steps = np.zeros(B, dtype=np.int64)  # chunks into each row's episode
        # (B,10,C,H,W), right-aligned: the last z_len[b] frames of row b are valid.
        self.z_hist = None
        self.z_len = np.zeros(B, dtype=np.int64)
        self.prev_img = torch.zeros(B, 3, 128, 128, device=device)
        self.curr_img = torch.zeros(B, 3, 128, 128, device=device)
        self.prev_state = torch.zeros(B, 4, device=device)
        self.curr_state = torch.zeros(B, 4, device=device)
        # Last commanded EE-xy target per row, for delta-action integration.
        self.last_target = np.zeros((B, 4), dtype=np.float64)
        self._templates, self._tc, self._frame0_area = [None] * B, [None] * B, [0] * B
        self.frame0_u8 = np.zeros((B, 128, 128, 3), dtype=np.uint8)
        self.angle_prev = torch.zeros(B, dtype=torch.float32)  # upright init => angle 0
        self._consec_reject = np.zeros(B, dtype=np.int64)  # for the stuck-prev recovery (see stuck_reject_limit)
        self._consec_bad = np.zeros(B, dtype=np.int64)  # for morph_terminate_patience (see early_stop_rule)
        # Cumulative diagnostic counters, kept across resets.
        for name in ("_jump_reject_count", "_lost_track_count", "_morph_reject_count",
                     "_stuck_recovery_count", "_early_stop_count"):
            setattr(self, name, getattr(self, name, 0))

        self._reset_metrics()
        self._reset_rows(np.arange(B))
        self._is_start = False
        return self._wrap_obs(), {}

    @torch.no_grad()
    def _reset_rows(self, rows):
        """Real-reset only `rows` (np int array) and re-initialize their per-row state;
        other rows are untouched. The caller resets these rows' episode metrics after
        it has read the finished episodes' values."""
        device = self.device
        imgs0, homes, frames0 = [], [], []
        for _ in rows:
            img, home, frame = self._one_real_reset()
            imgs0.append(img)
            homes.append(home)
            frames0.append(frame)
        f0 = torch.from_numpy(np.stack(imgs0)).to(device)  # (R,3,128,128) f32 [0,1]
        home_xy = np.stack(homes).astype(np.float32)  # (R,4)
        f0n = self.wm.normalizer["top_pov"].normalize(f0)
        z0 = self.wm.encoder_forward(f0n).to(self.wm.dtype)  # (R,C,H,W)
        if self.z_hist is None:
            self.z_hist = torch.zeros(
                self.num_envs, 10, *z0.shape[1:], dtype=z0.dtype, device=z0.device
            )
        idx = torch.as_tensor(rows, dtype=torch.long, device=self.z_hist.device)
        self.z_hist[idx] = 0
        self.z_hist[idx, -1] = z0
        self.z_len[rows] = 1

        idx_dev = torch.as_tensor(rows, dtype=torch.long, device=device)
        idx_cpu = torch.as_tensor(rows, dtype=torch.long)
        self.prev_img[idx_dev] = f0
        self.curr_img[idx_dev] = f0
        self.prev_state[idx_dev] = torch.from_numpy(home_xy).to(device)
        self.curr_state[idx_dev] = torch.from_numpy(home_xy).to(device)
        self.exec_actions_hist[idx_cpu] = 0.0
        self.t[rows] = 0
        self.steps[rows] = 0
        # Starts at the settled home pose, like real_sim_chunk_server.reset_row.
        self.last_target[rows] = home_xy.astype(np.float64)
        self.angle_prev[idx_cpu] = 0.0
        self._consec_reject[rows] = 0
        self._consec_bad[rows] = 0
        # Per-row template library (each row's own upright frame0 anchors its estimator).
        # thetas=REWARD_THETAS: full 360 deg grid, wider than the shared module default
        # -- see REWARD_THETAS's docstring for why (RL exploration drives the T past
        # the narrower collection-time grid's -130 deg edge).
        for i, b in enumerate(rows):
            tpl, tc = make_templates(frames0[i], thetas=REWARD_THETAS)
            self._templates[b] = tpl
            self._tc[b] = tc
            self._frame0_area[b] = int(red_mask(frames0[i]).sum())
            self.frame0_u8[b] = frames0[i]

    # ------------------------------------------------------------------ obs
    def _wrap_obs(self):
        """RLinf CNNPolicy obs contract, identical to IWSRotateTSimEnv._wrap_obs:
        main_images (B,128,128,3) in [0,255] (the current decoded frame only),
        states (B,state_dim). state_history "none" -> a constant zero state, i.e.
        an image-only policy. (The old {"image","agent_pos"} dict was the
        diffusion_policy convention and would KeyError inside CNNPolicy.)"""
        main_images = self.curr_img.float().permute(0, 2, 3, 1) * 255.0
        if self.state_history == "none":
            states = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        else:
            states = self.curr_state.float()
        return {"main_images": main_images, "states": states}

    # -------------------------------------------------------------- reward
    def _robust_angle_end(self, dec_u8_last3, row):
        """dec_u8_last3: list of up-to-3 uint8 (128,128,3) frames (most recent last),
        for row `row`. Returns (angle_end_rad, accepted: bool, morph_detected: bool,
        shrink_detected: bool).

        morph_detected specifically means "mask area grew past area_ratio_high" --
        the validated signature of the T deforming into a hallucinated blob (see
        area_ratio_high's definition in _build_dataset for the labeled-data evidence:
        area growth separates genuinely-morphed frames from merely-occluded-but-correct
        ones far better than IoU does). Callers (chunk_step) should terminate the
        episode when this fires -- once the WM has visibly hallucinated the object out
        of its correct shape, continuing the rollout wastes compute on an uninformative
        trajectory rather than getting a fresh one via reset."""
        templates, tc = self._templates[row], self._tc[row]
        f0_area = self._frame0_area[row]
        prev = float(self.angle_prev[row])

        raw = []  # (angle_or_None, iou, area) for ALL 3 frames, unfiltered -- for diagnostics
        reads = []  # (angle, iou) for frames that pass BOTH sanity checks
        any_mask_ok = False  # at least one frame had a plausible-area mask (area check passed)
        morph_detected = False  # at least one frame's area exceeded area_ratio_high
        # At least one frame's area fell below area_ratio_low, including no mask at all:
        # the T losing an arm (collapsing into an L/V shape) shrinks the mask, which
        # growth-only morph detection missed. Counts toward early stop only (user
        # decision 2026-09-13); the reward does not special-case it.
        shrink_detected = False
        for frame in dec_u8_last3:
            angle, iou, area = est_angle_with_conf(frame, templates, tc, thetas=REWARD_THETAS)
            raw.append((angle, iou, area))
            if area < self.area_ratio_low * f0_area:
                shrink_detected = True
            if angle is None:
                continue
            if area > self.area_ratio_high * f0_area:
                morph_detected = True
            # area sanity: reject if the mask ballooned/collapsed vs frame 0's own area
            # (fixed overhead camera -> the T's projected area shouldn't swing wildly
            # under pure rotation; a big deviation flags occlusion/color hallucination).
            # Bounds tightened from an original [0.3,3.0] to [area_ratio_low,
            # area_ratio_high] (default [0.75,1.20]) -- see _build_dataset for the
            # labeled-data evidence this was necessary (the loose bound missed 97.8%
            # of genuinely morphed frames in a validation set).
            if area < self.area_ratio_low * f0_area or area > self.area_ratio_high * f0_area:
                continue
            any_mask_ok = True
            # HARD IoU floor. See min_iou's definition (_build_dataset) for why this is
            # 0.35, not the originally-planned 0.5 -- direct inspection of rejected
            # frames in that 0.34-0.50 band showed legitimate, correctly-rotated T's
            # partly occluded by the gripper mechanism (present in every frame of this
            # task), not deformed shapes. Read is dropped entirely below the floor, not
            # just down-weighted, so a single badly-deformed read can't dominate the
            # median just because it's the only candidate in the window.
            if iou < self.min_iou:
                continue
            reads.append((angle, iou))

        if not reads:
            if any_mask_ok:
                self._morph_reject_count += 1  # had a plausible mask, but shape didn't match any rotation
            else:
                self._lost_track_count += 1  # no mask at all, or wildly wrong area
            self._consec_reject[row] += 1
            if hasattr(self, "_debug_log") and self._debug_log is not None:
                self._debug_log.append(dict(
                    row=row, kind=("morph" if any_mask_ok else "lost"),
                    raw=raw, f0_area=f0_area, prev_deg=np.degrees(prev),
                    morph_detected=morph_detected,
                ))
            return prev, False, morph_detected, shrink_detected  # carry forward, not accepted (prog=0 for this chunk)

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

        # Wraparound-safe angular distance -- REWARD_THETAS now spans the full circle
        # (see its docstring), so a naive difference would misreport e.g. -179 -> +179
        # deg (a genuine 2 deg step) as a spurious ~358 deg jump.
        jump_deg = abs(((np.degrees(candidate - prev) + 180) % 360) - 180)
        if jump_deg > self.jump_reject_deg:
            self._consec_reject[row] += 1
            # Stuck-prev recovery: don't let one stale `prev` (e.g. a transient
            # misread) poison the reward for the rest of the episode by making every
            # subsequent read "far from prev" too. We HAVE a plausible candidate here
            # (reads is non-empty) -- after enough consecutive rejects, trust it.
            if self.stuck_reject_limit > 0 and self._consec_reject[row] >= self.stuck_reject_limit:
                self._stuck_recovery_count += 1
                self._consec_reject[row] = 0
                if hasattr(self, "_debug_log") and self._debug_log is not None:
                    self._debug_log.append(dict(
                        row=row, kind="stuck_recovery", raw=raw, f0_area=f0_area,
                        prev_deg=np.degrees(prev), candidate_deg=np.degrees(candidate),
                        jump_deg=jump_deg,
                    ))
                return candidate, True, morph_detected, shrink_detected
            self._jump_reject_count += 1
            if hasattr(self, "_debug_log") and self._debug_log is not None:
                self._debug_log.append(dict(
                    row=row, kind="jump", raw=raw, f0_area=f0_area,
                    prev_deg=np.degrees(prev), candidate_deg=np.degrees(candidate),
                    jump_deg=jump_deg, morph_detected=morph_detected,
                ))
            return prev, False, morph_detected, shrink_detected

        self._consec_reject[row] = 0
        return candidate, True, morph_detected, shrink_detected

    # -------------------------------------------------------------- chunk_step
    @torch.no_grad()
    def chunk_step(self, actions):
        """actions arrive as (B, 1, 4*n_act) -- CNNPolicy emits exactly one flat
        vector per forward call (num_action_chunks stays 1), so
        actor.model.action_dim is set to 4*n_action_steps in the config and
        reshaped here into a genuine (B, n_act, 4) waypoint chunk, matching
        imagine_batch's own DP-chunk convention exactly (a real per-call
        `policy.predict_action(obs_dict)["action"]` is (B,n_act,4) too -- see
        collect_imagined_rotate_t.py's imagine_batch). Already prepared by
        rlinf.envs.action_utils.prepare_actions (pass-through for this env type --
        see rlinf_integration/ACTION_UTILS_PATCH.md).

        Returns ([obs], rewards (B,1), terminations (B,1), truncations (B,1), [infos])
        -- WanEnv.chunk_step's return contract (the BaseWorldEnv abstract-method
        docstring's "(obs, reward, done, info)" is NOT what env_worker expects), with
        the chunk axis width equal to actor.model.num_action_chunks=1."""
        B, device, n_act = self.num_envs, self.device, self.n_act
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32)
        actions = actions.detach().to("cpu", dtype=torch.float32).reshape(B, n_act, 4)
        if self.action_mode == "delta":
            # Normalized per-step deltas -> absolute EE-xy targets, integrated and
            # clipped exactly as real_sim_chunk_server.to_targets does; the WM keeps
            # receiving absolute targets, the only action form it was trained on.
            a = actions.numpy().astype(np.float64).clip(-1.0, 1.0) * self.max_step
            targets = np.empty_like(a)
            for b in range(B):
                t_b = self.last_target[b].copy()
                for k in range(n_act):
                    t_b = np.clip(t_b + a[b, k], self.target_lo, self.target_hi)
                    targets[b, k] = t_b
                self.last_target[b] = t_b
            actions = torch.from_numpy(targets.astype(np.float32))

        for b in range(B):
            t_b = int(self.t[b])
            self.exec_actions_hist[b, t_b : t_b + n_act] = actions[b]
            self.exec_actions_hist[b, t_b + n_act] = actions[b, -1]  # trailing pad

        # Rows are at different points in their own episodes (per-row reset), so their
        # WM context lengths differ. dynamics_forward depends only on that length and
        # the aligned action window (imagine_batch: frames lo..t, lo = max(0, t-9)),
        # and t has only three regimes -- 0, 8 and >=16 raw steps -- so rows are grouped
        # by min(t, 16) and each group runs exactly as a same-age imagine_batch would.
        z_new = None
        key = np.minimum(self.t, 16)
        for k in np.unique(key):
            rows = np.nonzero(key == k)[0]
            idx = torch.as_tensor(rows, dtype=torch.long, device=self.z_hist.device)
            L = min(int(k), 9) + 1
            assert (self.z_len[rows] == L).all(), (int(k), self.z_len[rows])
            act_win = torch.stack([
                self.exec_actions_hist[b, max(0, int(self.t[b]) - 9) : int(self.t[b]) + n_act + 1]
                for b in rows
            ]).to(device)
            act_win = self.wm.normalizer["action"].normalize(act_win).to(self.wm.dtype)
            z_group = self.wm.dynamics_forward(self.z_hist[idx, -L:], act_win)  # (G,n_act,C,H,W)
            if z_new is None:
                z_new = torch.empty(
                    B, *z_group.shape[1:], dtype=z_group.dtype, device=z_group.device
                )
            z_new[idx] = z_group
        self.z_hist = torch.cat([self.z_hist, z_new], dim=1)[:, -10:]
        self.z_len = np.minimum(self.z_len + n_act, 10)

        Bn = B * n_act
        dec = render_img_cm(
            self.wm, z_new.reshape(Bn, *z_new.shape[2:]), resolution=128,
            normalizer=self.wm.normalizer, num_views=1, batch_size=16,
        ).float().clamp(0, 1).reshape(B, n_act, 3, 128, 128)
        dec_u8 = (dec * 255).round().byte().permute(0, 1, 3, 4, 2).cpu().numpy()  # (B,n_act,128,128,3)

        self.prev_img = dec[:, -2] if n_act >= 2 else self.curr_img
        self.curr_img = dec[:, -1]
        self.t += n_act
        rows_all = torch.arange(B)
        t2 = torch.as_tensor(self.t)
        self.prev_state = self.exec_actions_hist[rows_all, t2 - 2].to(device)
        self.curr_state = self.exec_actions_hist[rows_all, t2 - 1].to(device)
        self.steps += 1

        k = min(3, n_act)
        rewards_last = torch.zeros(B, dtype=torch.float32, device=device)
        terminations_last = torch.zeros(B, dtype=torch.bool, device=device)
        early_stop = torch.zeros(B, dtype=torch.bool, device=device)
        for b in range(B):
            last_frames = [dec_u8[b, j] for j in range(n_act - k, n_act)]
            angle_end, _accepted, morph_detected, shrink_detected = self._robust_angle_end(last_frames, b)
            if morph_detected and _accepted and hasattr(self, "_debug_log") and self._debug_log is not None:
                # The reject path already logs a "morph" debug_log entry itself; this
                # covers the accept path (a neighbor frame in the window was valid, but
                # another frame in the same window still showed hallucinated growth).
                self._debug_log.append(dict(
                    row=b, kind="morph_terminate_on_accept", raw=[],
                    prev_deg=np.degrees(float(self.angle_prev[b])),
                    candidate_deg=np.degrees(angle_end),
                ))
            # On a morph-detected chunk, the reward is exactly the per-step time
            # penalty -0.01 -- no progress credit, no success bonus. angle_end
            # here is either a carried-forward stale `prev` (reject path) or a
            # neighbor-frame reading from the same window that happened to pass
            # (accept path); neither is a trustworthy basis for crediting
            # progress OR declaring success once the mask itself has shown
            # hallucinated growth in this window. angle_prev is also NOT
            # advanced to this untrustworthy angle_end -- freezing it at the
            # last trustworthy value prevents a spurious progress spike (or
            # dip) on the NEXT good chunk, which would otherwise measure
            # "progress" against a baseline that was itself hallucinated.
            if morph_detected:
                prog = 0.0
                success = False
            else:
                prog = float(self.angle_prev[b]) - angle_end
                success = bool(angle_end <= np.radians(self.terminal_deg))
                self.angle_prev[b] = angle_end
            rewards_last[b] = prog * 5.0 - 0.01 + (5.0 if success else 0.0)
            terminations_last[b] = success
            # Patience, not immediate truncation -- see morph_terminate_patience's
            # definition (_build_dataset) for the full-pool evidence this was
            # necessary (immediate truncation discarded ~43% of episodes for
            # single-chunk artifacts that self-correct ~99% of the time). The
            # reward/success suppression above still fires on ANY morph_detected
            # chunk regardless of patience -- that's a separate, still-correct
            # safety property (never credit progress or success from an
            # untrustworthy read, even while the episode is still allowed to
            # continue).
            if self.early_stop_rule == "any_reject":
                bad = morph_detected or shrink_detected or not _accepted
            else:
                bad = morph_detected
            self._consec_bad[b] = self._consec_bad[b] + 1 if bad else 0
            early_stop[b] = bool(self._consec_bad[b] >= self.morph_terminate_patience)
        if early_stop.any():
            self._early_stop_count += int(early_stop.sum())

        # Per-row truncation: the row's own step budget, OR early stop (patience
        # consecutive bad chunks). Both are truncations, not terminations, so the
        # critic still bootstraps from the final frame -- a user decision
        # (2026-09-13) kept as an ablation candidate: terminating on early stop would
        # instead zero the value past a persistent hallucination.
        horizon_cut = torch.as_tensor(self.steps >= self.wm_max_chunks, device=device)
        truncations_last = horizon_cut | early_stop

        # Width 1 along the chunk axis (= actor.model.num_action_chunks): the whole
        # 8-waypoint chunk is one flat action. env_worker's bootstrap placeholder is
        # (B, num_action_chunks), so (B, n_act) tensors crash torch.stack in the
        # trajectory builder -- the same failure the sim env hit.
        chunk_rewards = rewards_last.unsqueeze(-1)
        chunk_terminations = terminations_last.unsqueeze(-1)
        chunk_truncations = truncations_last.unsqueeze(-1)

        past_dones = terminations_last | truncations_last
        pre_reset_obs = self._wrap_obs()
        infos = {}
        infos = self._record_metrics(rewards_last, terminations_last, infos)
        # See IWSRotateTSimEnv.chunk_step's identical line for why: surfaces
        # BaseWorldEnv's own already-tracked self.success_once into
        # infos["episode"] so it flows through to eval/success (a real success
        # RATE, not just eyeballed from eval/return's magnitude) the same way
        # "return" already does -- no RLinf-side (external repo) code touched.
        infos["episode"]["success"] = self.success_once.clone().float()
        infos["jump_reject_count"] = self._jump_reject_count
        infos["lost_track_count"] = self._lost_track_count
        infos["morph_reject_count"] = self._morph_reject_count
        infos["stuck_recovery_count"] = self._stuck_recovery_count
        infos["early_stop_count"] = self._early_stop_count
        # Why each finished row ended (env_worker keeps only done rows' values),
        # logged as env/early_stop and env/horizon_cut next to env/success.
        infos["episode"]["early_stop"] = early_stop.float()
        infos["episode"]["horizon_cut"] = (horizon_cut & ~terminations_last & ~early_stop).float()

        if past_dones.any():
            # Per-row reset: only finished rows get a new real reset; the others keep
            # running. env_worker.py indexes infos["final_info"] and reads
            # infos["final_observation"] as next_obs whenever any row is done (it
            # asserts next_obs is not None) -- both crashed earlier runs when missing.
            # dict(infos), not infos: a self-referential dict crashes with
            # RecursionError in EnvOutput's nested device walk (as in the sim env).
            infos["final_observation"] = pre_reset_obs
            infos["final_info"] = dict(infos)
            infos["_final_info"] = past_dones
            infos["_final_observation"] = past_dones
            done_rows = torch.nonzero(past_dones).squeeze(-1).cpu().numpy()
            self._reset_rows(done_rows)
            self._reset_metrics(env_idx=torch.as_tensor(done_rows, dtype=torch.long, device=device))
            extracted_obs = self._wrap_obs()
        else:
            extracted_obs = pre_reset_obs

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

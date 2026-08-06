"""Lockstep teleop demo: MuJoCo PushT sim vs. the IWS latent world model.

Drives the real MuJoCo/ALOHA PushT simulator and the trained latent world model from
the *same* keyboard-commanded action stream, renders both plus their difference, and
reports PSNR and T-rotation drift live in the browser.

The world model runs pure open-loop: it sees the ground-truth frame exactly once, at
reset (or when the operator presses Re-sync), and every later frame is its own
imagination. Divergence between the two panes is therefore the quantity of interest.

Run with ``bash deploy/start_sim_demo.sh`` (which sets ``MUJOCO_GL=egl``), or exercise
the engine headlessly with ``python deploy/sim_demo_server.py --ticks 100``.
"""

import os

# Must precede any mujoco / dm_control / gym_aloha import: without it dm_control falls
# back to glfw and dies on the missing DISPLAY over SSH.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
# scripts/ is not a package; its modules are imported by path, as eval_dp_rotate_t.py
# and friends do. The repo root itself is needed so `deploy.` resolves when this file is
# run as a script rather than imported by uvicorn.
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "data_collection"))

import collect_rotate_t as C  # noqa: E402
import cv2  # noqa: E402
import eval_metric_correlation as M  # noqa: E402
import gym_aloha.env as gae  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import torch  # noqa: E402
from fastapi import (  # noqa: E402
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from gym_aloha.env import AlohaEnv  # noqa: E402
from sim_aloha_dataset_collection_scripted import (  # noqa: E402
    generate_random_init_action,
    get_current_arm_positions,
    trajectory_to_joint_actions,
)
from yixuan_utilities.draw_utils import center_crop  # noqa: E402
from yixuan_utilities.kinematics_helper import KinHelper  # noqa: E402

from deploy.sim_demo_metrics import (  # noqa: E402
    diff_image,
    est_angle,
    make_templates,
    mse,
    psnr,
    wrap_angle,
)
from interactive_world_sim.algorithms.common.diffusion_helper import (  # noqa: E402
    render_img_cm,
)
from interactive_world_sim.utils.pose_utils import PoseType, pose_convert  # noqa: E402

# Controller gains, matching data collection and eval exactly
# (lifted from scripts/eval_dp_rotate_t.py:56).
DT, K_P, K_V, ACC_LIM, VEL_LIM = 1 / 10.0, 50, 10, 10.0, 0.04

RESOLUTION = 128
HIST = 10  # latent-history window the dynamics conditions on
FRAME_DT = 0.1  # 10 Hz, the control rate the world model was trained at
STALE_S = 0.5  # no client message for this long => freeze (dropped-keyup guard)
DEFAULT_CKPT = "ckpts/push_t/epoch=3-step=90000.ckpt"
DEFAULT_SEED = 900_000  # disjoint from the eval seed 7000 and collector base 500_000

# Keyboard -> world-frame [dx_left, dy_left, dx_right, dy_right]. The top_pov camera is
# a true top-down view, so this is the identity map (cf. the pusht branch of
# deploy/server.py:299-309). If a direction feels inverted, flip a sign here.
KEY_AXES: Tuple[Tuple[str, int, float], ...] = (
    ("d", 0, +1.0),
    ("a", 0, -1.0),
    ("w", 1, +1.0),
    ("s", 1, -1.0),
    ("l", 2, +1.0),
    ("j", 2, -1.0),
    ("i", 3, +1.0),
    ("k", 3, -1.0),
)

# The stock random-theta sampler, captured before anything monkeypatches it. `wm_train`
# init mode restores this; the other modes swap in an upright sampler.
_STOCK_SAMPLE_PUSHT_POSE = gae.sample_pusht_pose

# The checkpoint carries ~344 unused validation_fvd_model.detector.* I3D tensors, which
# we load with strict=False on purpose. Lightning warns with every key name, burying the
# real startup output in hundreds of lines -- unusable when run over ssh in a terminal.
# This goes through warnings.warn, not logging, so a logger level would not touch it.
warnings.filterwarnings(
    "ignore", message=".*Found keys that are not in the model state dict.*"
)


@dataclass
class TickResult:
    """One rendered step: the JSON header fields plus the three frames to encode."""

    header: Dict[str, Any]
    frames: List[Tuple[str, npt.NDArray[np.uint8]]]


class LockstepRunner:
    """Drives the MuJoCo PushT sim and the world model from one shared action stream.

    All heavy work (MuJoCo, CUDA, OpenCV) happens inside this object's methods, which
    are called from a single dedicated worker thread. Nothing here is re-entrant.
    """

    def __init__(
        self,
        ckpt: str = DEFAULT_CKPT,
        config_name: str = "pusht_mujoco",
        device: str = "cuda:0",
        seed: int = DEFAULT_SEED,
        init_mode: str = "fixed",
        action_lag: int = 1,
        img_format: str = "png",
        dec_infer_steps: int = 2,
        settle_steps: int = 100,
        arm_spread: float = 0.06,
    ) -> None:
        """Load the world model and build the simulator. Does not reset yet."""
        if action_lag not in (0, 1):
            raise ValueError(f"action_lag must be 0 or 1, got {action_lag}")
        cv2.setNumThreads(1)

        self.device = torch.device(device)
        cfg = M.load_viz_cfg(config_name)
        self.wm = M.load_model(str(_REPO_ROOT / ckpt), cfg.algorithm, str(self.device))
        # 2, not the 1 used by the offline batch scripts. Measured here: the decoder's
        # own reconstruction PSNR jumps 27 -> 45 dB (i.e. becomes visually lossless) for
        # +3 ms/tick, so the reported PSNR measures dynamics error rather than decoder
        # blur. 3 and 5 steps buy nothing further.
        self.wm.dec_infer_steps = dec_infer_steps

        self.env = AlohaEnv("pusht")
        self.kin = KinHelper(robot_name="trossen_vx300s")

        self.seed = seed
        self.init_mode = init_mode
        self.action_lag = action_lag
        self.img_format = img_format
        self.settle_steps = settle_steps
        self.arm_spread = arm_spread
        self.episode = 0
        self.resyncs = 0

        # Keyboard delta scale, in normalized action units. Precomputed once here rather
        # than per tick as deploy/server.py:404-421 does. The 1/100 factor gives ~3 mm
        # of travel per 10 Hz step, comfortably inside VEL_LIM=0.04.
        stats = self.wm.normalizer["action"].state_dict()
        rng = (
            stats["params_dict.input_stats.max"] - stats["params_dict.input_stats.min"]
        )
        self.delta_scale = (1.0 / (100.0 * (rng / rng.max()))).cpu().numpy()

        # Populated by reset().
        self.world_t_bases = np.zeros((2, 4, 4))
        self.curr_vel = np.zeros(6)
        self.act_hist: List[torch.Tensor] = []
        self.z_hist = torch.zeros(0)
        self.templates = np.zeros(0, dtype=bool)
        self.tc: Tuple[float, float] = (0.0, 0.0)
        self.theta0 = 0.0
        self.tick = 0
        self.psnr_sum = 0.0
        self.psnr_n = 0
        self.psnr_floor = 0.0
        self.gt_u8 = np.zeros((RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
        self.prev_gt_u8 = self.gt_u8
        self.wm_u8 = self.gt_u8

    # ------------------------------------------------------------------ env access --
    def _task_obs(self) -> Dict[str, Any]:
        """Raw dm_control observation dict (qpos, env_state, images, base poses)."""
        return self.env._env.task.get_observation(self.env._env.physics)  # noqa: SLF001

    def _get_obs(
        self,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        """Build the exact training observation: cropped 128x128 top_pov + EE-XY."""
        # lifted from scripts/eval_dp_rotate_t.py:106-114
        obs = self._task_obs()
        crop = center_crop(obs["images"]["top_pov"], (RESOLUTION, RESOLUTION))
        rz = cv2.resize(crop, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
        image = np.moveaxis(rz, -1, 0).astype(np.float32) / 255.0
        ee = get_current_arm_positions(obs, self.kin, self.world_t_bases)
        return image, ee.astype(np.float32), rz

    def _servo(self, target_xy: npt.NDArray[np.float64], n: int) -> None:
        """Hold a fixed EE-XY target for n control steps (used by the wm_train init)."""
        for _ in range(n):
            joint, _ = trajectory_to_joint_actions(
                target_xy,
                self.world_t_bases,
                self.kin,
                self._task_obs()["qpos"][:14],
                self.curr_vel,
                DT,
                K_P,
                K_V,
                ACC_LIM,
                VEL_LIM,
            )
            self.env.step(joint)

    def _spread_arms(self, distance: float) -> None:
        """Drive both grippers outward from the centre of the table.

        At the env's reset pose the EEs sit at x = -+0.17 with the fingertips offset
        ~0.08 inward, so they end up at -+0.09 -- right on the edge of a T spawned at
        the origin, which reads as the block resting on the grippers. Moving them out
        keeps the whole T visible and unobstructed at the start of an episode.

        It also matters physically, not just visually: with the arms at home the block
        comes to rest at z=+0.017, i.e. propped on the grippers rather than on the
        table. Once they move clear it settles to z=-0.001. (The reference protocol,
        collect_rotate_t.settle_arms, shows the same effect at z=+0.006.)

        The reachable spread is bounded by the workspace clip in
        trajectory_to_joint_actions, so values beyond ~0.06 have no further effect.
        """
        if distance <= 0:
            return
        obs = self._task_obs()
        home = get_current_arm_positions(obs, self.kin, self.world_t_bases)
        # -x is outward for the left arm, +x for the right
        # (cf. collect_rotate_t.settle_arms).
        target = home + np.array([-distance, 0.0, distance, 0.0])
        self._servo(target, 40)

    # ------------------------------------------------------------- model plumbing --
    def _norm_action(self, ee: npt.NDArray[np.float32]) -> torch.Tensor:
        """Raw EE-XY in metres -> normalized (1, 4) action tensor clamped to [-1, 1]."""
        t = torch.from_numpy(ee.astype(np.float32))[None].to(self.device)
        return self.wm.normalizer["action"].normalize(t).clamp(-1.0, 1.0)

    def _unnorm(self, a_norm: torch.Tensor) -> npt.NDArray[np.float64]:
        """Normalized (1, 4) action tensor -> raw EE-XY in metres."""
        raw = self.wm.normalizer["action"].unnormalize(a_norm.float())
        return raw[0].cpu().numpy().astype(np.float64)

    def _encode(self, image: npt.NDArray[np.float32]) -> torch.Tensor:
        """(3, 128, 128) float [0,1] frame -> (1, 1, C, H, W) latent history."""
        t = torch.from_numpy(image)[None].to(self.device)
        z = self.wm.encoder_forward(self.wm.normalizer["top_pov"].normalize(t))
        return z.to(self.wm.dtype).unsqueeze(1)

    def _decode(self, z: torch.Tensor) -> npt.NDArray[np.uint8]:
        """(1, C, H, W) latent -> (128, 128, 3) uint8 RGB frame."""
        img = render_img_cm(
            self.wm,
            z,
            RESOLUTION,
            normalizer=self.wm.normalizer,
            num_views=1,
            batch_size=16,
        )
        arr = img.float().clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
        return (arr * 255).round().astype(np.uint8)

    # --------------------------------------------------------------------- public --
    def warmup(self) -> None:
        """Pay the cuDNN / EGL first-call cost before the port binds."""
        try:
            self.reset(None, self.init_mode, 3.0)
        except Exception as exc:  # pragma: no cover - startup diagnostics
            raise RuntimeError(
                "warmup failed. If this is a rendering error, MUJOCO_GL=egl must be "
                "set before import -- use bash deploy/start_sim_demo.sh."
            ) from exc
        delta = np.zeros(4)
        delta[0] = self.delta_scale[0]
        self.step(delta, 3.0)
        self.reset(None, self.init_mode, 3.0)

    @torch.no_grad()
    def reset(
        self, seed: Optional[int], init_mode: str, diff_gain: float
    ) -> TickResult:
        """Reset sim and world model to a fresh, mutually consistent episode."""
        self.init_mode = init_mode
        self.seed = self.seed + 1 if seed is None else int(seed)

        # The pose sampler is a module global read inside env.reset, so patch first.
        if init_mode == "fixed":
            gae.sample_pusht_pose = C.make_upright_pose_sampler((0.0, 0.0), (0.0, 0.0))
        elif init_mode == "random":
            gae.sample_pusht_pose = C.make_upright_pose_sampler(
                (-0.08, 0.08), (-0.08, 0.08)
            )
        elif init_mode == "wm_train":
            gae.sample_pusht_pose = _STOCK_SAMPLE_PUSHT_POSE
        else:
            raise ValueError(f"unknown init_mode {init_mode!r}")

        while True:
            np.random.seed(self.seed)
            self.env.reset(seed=self.seed)
            obs = self._task_obs()
            self.world_t_bases = np.stack(
                [
                    pose_convert(
                        obs["left_base"][None], PoseType.POS_QUAT, PoseType.MAT
                    )[0],
                    pose_convert(
                        obs["right_base"][None], PoseType.POS_QUAT, PoseType.MAT
                    )[0],
                ]
            )
            self.curr_vel = np.zeros(6)

            if init_mode == "wm_train":
                # Matches sim_aloha_dataset_collection_scripted.py:task_reset -- the
                # protocol that actually produced this checkpoint's training data. The
                # 100-step servo also gives the T time to fall and settle.
                self._servo(generate_random_init_action(self.world_t_bases), 100)
                break
            # Long enough for the T to fall from its 0.07 spawn height and come to
            # rest before anything is observed or encoded.
            C.stabilize_t(self.env, n=self.settle_steps)
            if init_mode == "fixed":
                self._spread_arms(self.arm_spread)
                break
            C.settle_arms(
                self.env,
                self.world_t_bases,
                self.kin,
                np.zeros(6),
                DT,
                K_P,
                K_V,
                ACC_LIM,
                VEL_LIM,
            )
            if abs(C.t_angle(self.env)) <= C.UPRIGHT_TOL:
                break
            self.seed += 1  # the settle bumped the T off upright -- fresh reset

        image, ee, frame0 = self._get_obs()
        self.act_hist = [self._norm_action(ee)]
        self.z_hist = self._encode(image)
        self.theta0 = C.t_angle(self.env)
        self.templates, self.tc = make_templates(frame0)

        self.tick = 0
        self.psnr_sum = 0.0
        self.psnr_n = 0
        self.episode += 1
        self.resyncs = 0
        self.gt_u8 = self.prev_gt_u8 = frame0
        # Decode z_0 and show it as the world model's frame 0: this is the decoder's
        # reconstruction ceiling, which no later tick can beat.
        self.wm_u8 = self._decode(self.z_hist[:, -1])
        self.psnr_floor = psnr(frame0, self.wm_u8)
        return self._payload("reset", {}, diff_gain, unreachable=False)

    @torch.no_grad()
    def resync(self, diff_gain: float) -> TickResult:
        """Re-ground the world model on the current sim frame, leaving the sim alone."""
        image, _, _ = self._get_obs()
        self.z_hist = self._encode(image)
        self.act_hist = [
            self.act_hist[-1]
        ]  # keep the invariant from a length-1 history
        self.wm_u8 = self._decode(self.z_hist[:, -1])
        self.resyncs += 1
        return self._payload("reset", {}, diff_gain, unreachable=False)

    @torch.no_grad()
    def step(self, delta: npt.NDArray[np.float64], diff_gain: float) -> TickResult:
        """Advance sim and world model by one 10 Hz control step on the same action."""
        timing: Dict[str, float] = {}
        t_start = time.perf_counter()

        a_prev = self.act_hist[-1]
        d = torch.from_numpy(delta.astype(np.float32))[None].to(self.device)
        a_k = torch.clamp(a_prev + d, -1.0, 1.0)
        self.act_hist.append(a_k)
        clamped = bool(torch.any(torch.abs(a_prev + d) > 1.0).item())

        # The training action channel is obs/ee_pos (sim_aloha_dataset.py:97-102), i.e.
        # the *achieved* EE in frame k, which argues for lag 0. But the PID needs a step
        # to reach a commanded target, and lag 1 measured consistently better here
        # (mean PSNR over 40 ticks: 22.88 vs 22.42 wm_train, 18.58 vs 18.44 fixed), so
        # lag 1 is the default. Kept as a knob because the margin is small.
        cmd = a_k if self.action_lag == 0 else a_prev
        target_m = self._unnorm(cmd)

        t0 = time.perf_counter()
        joint, target_clip = trajectory_to_joint_actions(
            target_m,
            self.world_t_bases,
            self.kin,
            self._task_obs()["qpos"][:14],
            self.curr_vel,  # persistent: mutated in place, carries velocity continuity
            DT,
            K_P,
            K_V,
            ACC_LIM,
            VEL_LIM,
        )
        self.env.step(joint)
        _, ee_m, gt_u8 = self._get_obs()
        gt_angle = wrap_angle(C.t_angle(self.env) - self.theta0)
        unreachable = bool(np.linalg.norm(target_clip - target_m) > 1e-3)
        timing["sim"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        n_hist = int(self.z_hist.shape[1])
        act_win = torch.cat(self.act_hist[-(n_hist + 1) :], dim=0)[None]
        assert act_win.shape[1] == n_hist + 1, (act_win.shape, self.z_hist.shape)
        z_new = self.wm.dynamics_forward(self.z_hist, act_win.to(self.wm.dtype))
        self.z_hist = torch.cat([self.z_hist, z_new], dim=1)[:, -HIST:]
        timing["dyn"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        wm_u8 = self._decode(z_new[:, -1])
        timing["dec"] = (time.perf_counter() - t0) * 1e3

        self.tick += 1
        self.prev_gt_u8 = self.gt_u8
        self.gt_u8 = gt_u8
        self.wm_u8 = wm_u8

        action = {
            "norm": [round(v, 4) for v in a_k[0].cpu().numpy().tolist()],
            "target_m": [round(v, 4) for v in target_m.tolist()],
            "ee_m": [round(float(v), 4) for v in ee_m.tolist()],
            "gt_angle": gt_angle,
            "clamped": clamped,
        }
        del t_start  # `total` is summed from the stages in send_tick, once enc is known
        return self._payload("tick", timing, diff_gain, unreachable, action)

    # -------------------------------------------------------------------- payload --
    def _payload(
        self,
        kind: str,
        timing: Dict[str, float],
        diff_gain: float,
        unreachable: bool,
        action: Optional[Dict[str, Any]] = None,
    ) -> TickResult:
        """Assemble the JSON header and the three frames for one message."""
        t0 = time.perf_counter()
        cur_psnr = psnr(self.gt_u8, self.wm_u8)
        if kind == "tick":
            self.psnr_sum += cur_psnr
            self.psnr_n += 1
        wm_angle, t_pixels = est_angle(self.wm_u8, self.templates, self.tc)
        gt_angle = (
            action["gt_angle"]
            if action is not None
            else wrap_angle(C.t_angle(self.env) - self.theta0)
        )
        drift = (
            None if wm_angle is None else np.degrees(wrap_angle(wm_angle - gt_angle))
        )
        timing["metric"] = round((time.perf_counter() - t0) * 1e3, 1)

        header: Dict[str, Any] = {
            "type": kind,
            "episode": self.episode,
            "seed": self.seed,
            "tick": self.tick,
            "resyncs": self.resyncs,
            "init_mode": self.init_mode,
            "action_lag": self.action_lag,
            "timing_ms": {k: round(v, 1) for k, v in timing.items()},
            "action": {k: v for k, v in (action or {}).items() if k != "gt_angle"},
            "metrics": {
                "psnr_db": round(cur_psnr, 2),
                "psnr_db_mean": (
                    round(self.psnr_sum / self.psnr_n, 2) if self.psnr_n else None
                ),
                "psnr_db_floor": round(self.psnr_floor, 2),
                "psnr_db_static": round(psnr(self.gt_u8, self.prev_gt_u8), 2),
                "mse": round(mse(self.gt_u8, self.wm_u8), 1),
                "angle_gt_deg": round(float(np.degrees(gt_angle)), 1),
                "angle_wm_deg": (
                    None if wm_angle is None else round(float(np.degrees(wm_angle)), 1)
                ),
                "angle_drift_deg": None if drift is None else round(float(drift), 1),
                "t_visible": wm_angle is not None,
                "t_pixels": t_pixels,
            },
        }
        header["action"]["unreachable"] = unreachable
        # `total` is the sum of the stages measured so far; send_tick folds in `enc`.
        header["timing_ms"]["total"] = round(sum(header["timing_ms"].values()), 1)
        frames = [
            ("gt", self.gt_u8),
            ("wm", self.wm_u8),
            ("diff", diff_image(self.gt_u8, self.wm_u8, diff_gain)),
        ]
        return TickResult(header=header, frames=frames)


# ------------------------------------------------------------------ wire framing --
def encode_image(frame: npt.NDArray[np.uint8], fmt: str) -> Tuple[str, bytes]:
    """Encode an RGB frame. PNG level 1 halves level 0's bytes for ~0.5 ms more."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if fmt == "jpeg":
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        mime = "image/jpeg"
    else:
        ok, buf = cv2.imencode(".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        mime = "image/png"
    if not ok:
        raise RuntimeError(f"failed to encode frame as {fmt}")
    return mime, buf.tobytes()


def pack_frame(header: Dict[str, Any], images: List[Tuple[str, str, bytes]]) -> bytes:
    """uint32 BE header length | UTF-8 JSON header | concatenated image payloads."""
    metas, blobs, off = [], [], 0
    for name, mime, data in images:
        metas.append({"name": name, "mime": mime, "off": off, "len": len(data)})
        blobs.append(data)
        off += len(data)
    hb = json.dumps({**header, "images": metas}).encode("utf-8")
    return struct.pack(">I", len(hb)) + hb + b"".join(blobs)


def unpack_frame(buf: bytes) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    """Inverse of pack_frame. Used by the self-test; the browser does this in JS."""
    (hlen,) = struct.unpack(">I", buf[:4])
    header = json.loads(buf[4 : 4 + hlen].decode("utf-8"))
    base = 4 + hlen
    images = {
        im["name"]: buf[base + im["off"] : base + im["off"] + im["len"]]
        for im in header["images"]
    }
    return header, images


# --------------------------------------------------------------------- web layer --
@dataclass
class Inbox:
    """Last-write-wins register for client messages. Never touches CUDA or MuJoCo."""

    keys: Dict[str, int] = field(default_factory=dict)
    seq: int = 0
    t_client: float = 0.0
    last_msg_t: float = field(default_factory=time.perf_counter)
    pending_reset: bool = False
    reset_seed: Optional[int] = None
    pending_resync: bool = False
    init_mode: str = "fixed"
    diff_gain: float = 3.0
    speed: float = 1.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def absorb(self, msg: Dict[str, Any]) -> None:
        """Fold one client message into the register."""
        self.last_msg_t = time.perf_counter()
        kind = msg.get("type", "action")
        if kind == "action":
            self.keys = msg.get("action", {}) or {}
            self.seq = int(msg.get("seq", 0))
            self.t_client = float(msg.get("t_client", 0.0))
        elif kind == "reset":
            self.pending_reset = True
            seed = msg.get("seed")
            self.reset_seed = None if seed is None or seed == "" else int(seed)
            self.keys = {}
        elif kind == "resync":
            self.pending_resync = True
        if msg.get("init_mode"):
            self.init_mode = str(msg["init_mode"])
        if "diff_gain" in msg:
            self.diff_gain = float(msg["diff_gain"])
        if "speed" in msg:
            self.speed = float(msg["speed"])


def parse_delta(
    keys: Dict[str, int], scale: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Held keys -> a normalized-action delta vector (4,)."""
    # lifted from the pusht branch of deploy/server.py:251-269
    delta = np.zeros(4)
    for key, axis, sign in KEY_AXES:
        if keys.get(key):
            delta[axis] = sign
    return delta * scale


EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="iws-sim")
RUNNER: Optional[LockstepRunner] = None
BUSY = asyncio.Lock()
# Bumped on every new connection. An older session sees its token go stale and exits,
# so reloading the tab always takes over rather than being refused -- a refusal is
# invisible in the browser (the panes just stay black) and is never what you want from
# a single-user local demo.
SESSION = 0
_ROLLOUT_DIR = _REPO_ROOT / "outputs" / "scaling_rollouts"
_WM_GALLERY_DIR = _REPO_ROOT / "outputs" / "wm_gallery"
_METRICS_JSON = _REPO_ROOT / "outputs" / "random_init_evals" / "scaling_metrics.json"
_FIGURES = {
    "success_rate": _REPO_ROOT / "outputs" / "scaling_plot_final.png",
    "rotation_deg": _REPO_ROOT / "outputs" / "scaling_plot_rotation.png",
    "wm_quality": _REPO_ROOT / "outputs" / "wm_quality_plot.png",
}


def _runner_from_env() -> LockstepRunner:
    """Build the runner from IWS_DEMO_* env vars (uvicorn imports us, so no argv)."""
    return LockstepRunner(
        ckpt=os.environ.get("IWS_DEMO_CKPT", DEFAULT_CKPT),
        device=os.environ.get("IWS_DEMO_DEVICE", "cuda:0"),
        seed=int(os.environ.get("IWS_DEMO_SEED", DEFAULT_SEED)),
        init_mode=os.environ.get("IWS_DEMO_INIT", "fixed"),
        action_lag=int(os.environ.get("IWS_DEMO_ACTION_LAG", 1)),
        img_format=os.environ.get("IWS_DEMO_IMG", "png"),
        dec_infer_steps=int(os.environ.get("IWS_DEMO_DEC_STEPS", 2)),
        settle_steps=int(os.environ.get("IWS_DEMO_SETTLE_STEPS", 100)),
        arm_spread=float(os.environ.get("IWS_DEMO_ARM_SPREAD", 0.06)),
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the model and warm the CUDA/EGL paths before the port accepts traffic."""
    global RUNNER
    t0 = time.perf_counter()
    RUNNER = await asyncio.get_running_loop().run_in_executor(EXEC, _runner_from_env)
    await asyncio.get_running_loop().run_in_executor(EXEC, RUNNER.warmup)
    print(f"[sim_demo] runner ready in {time.perf_counter() - t0:.1f}s", flush=True)
    yield


app = FastAPI(lifespan=lifespan)
# The UI is served from the user's own machine (iws-demo-frontend), so every request
# here is cross-origin. This binds to loopback and is reached through an SSH tunnel,
# so a permissive policy is appropriate.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> JSONResponse:
    """Identify the service. The UI lives in the separate iws-demo-frontend repo."""
    return JSONResponse(
        {
            "service": "interactive-world-sim backend",
            "ui": "run ./serve.py in the iws-demo-frontend repo",
            "endpoints": ["/ws", "/api/scaling", "/rollouts/...", "/figures/{name}"],
            "rollout_conditions": len(list(_ROLLOUT_DIR.glob("r*i*"))),
        }
    )


@app.get("/api/scaling")
def scaling_data() -> JSONResponse:
    """Published per-condition metrics merged with whatever rollouts exist so far.

    Read fresh on every request so the page picks up new conditions while
    `scripts/collect_scaling_rollouts.py` is still running.
    """
    metrics = json.loads(_METRICS_JSON.read_text()) if _METRICS_JSON.exists() else {}
    manifest_path = _ROLLOUT_DIR / "manifest.json"
    rollouts = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    gallery_path = _WM_GALLERY_DIR / "manifest.json"
    gallery = json.loads(gallery_path.read_text()) if gallery_path.exists() else {}
    return JSONResponse(
        {
            "metrics": metrics,
            "rollouts": rollouts.get("conditions", {}),
            "rollout_protocol": rollouts.get("protocol", {}),
            "wm_gallery": gallery,
        }
    )


@app.get("/figures/{name}")
def figure(name: str) -> FileResponse:
    """Serve one of the original published PNGs, for side-by-side comparison."""
    path = _FIGURES.get(name)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail=f"unknown figure {name!r}")
    return FileResponse(path, media_type="image/png")


_ROLLOUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/rollouts", StaticFiles(directory=str(_ROLLOUT_DIR)), name="rollouts")
_WM_GALLERY_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/wm_gallery", StaticFiles(directory=str(_WM_GALLERY_DIR)), name="wm_gallery")


async def send_tick(ws: WebSocket, res: TickResult, seq: int, t_client: float) -> None:
    """Encode and send one tick, with backpressure so stale frames never queue up."""
    assert RUNNER is not None
    t0 = time.perf_counter()
    images = [(name, *encode_image(fr, RUNNER.img_format)) for name, fr in res.frames]
    res.header["seq"] = seq
    res.header["t_client"] = t_client
    timing = res.header["timing_ms"]
    timing["enc"] = round((time.perf_counter() - t0) * 1e3, 1)
    timing["total"] = round(timing["total"] + timing["enc"], 1)
    await asyncio.wait_for(ws.send_bytes(pack_frame(res.header, images)), timeout=1.0)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    """Run the 10 Hz lockstep loop, taking over from any older connection."""
    global SESSION
    await ws.accept()
    assert RUNNER is not None
    SESSION += 1
    my_session = SESSION
    # Any older session sees the token change and breaks out within one tick, releasing
    # BUSY; this then acquires it. Serialising on BUSY keeps two sessions from driving
    # the one simulator concurrently.
    async with BUSY:
        inbox = Inbox(init_mode=RUNNER.init_mode)
        loop = asyncio.get_running_loop()

        async def recv_loop() -> None:
            try:
                while True:
                    msg = await ws.receive_json()
                    async with inbox.lock:
                        inbox.absorb(msg)
            except Exception:
                return  # disconnect or bad frame; the tick loop notices via done()

        recv_task = asyncio.create_task(recv_loop())
        try:
            await ws.send_json({"type": "resetting"})
            res = await loop.run_in_executor(
                EXEC, RUNNER.reset, None, inbox.init_mode, inbox.diff_gain
            )
            await send_tick(ws, res, 0, 0.0)

            next_t = time.perf_counter()
            while True:
                now = time.perf_counter()
                if now < next_t:
                    await asyncio.sleep(next_t - now)
                # No catch-up burst after a long freeze.
                next_t = max(next_t + FRAME_DT, time.perf_counter())

                # A frozen tick sends nothing, so the socket alone would never reveal a
                # dropped client -- and the BUSY lock would be held forever. recv_loop
                # finishing is the reliable disconnect signal.
                if recv_task.done():
                    break
                if my_session != SESSION:
                    await ws.close(code=1012, reason="superseded by a newer tab")
                    break

                async with inbox.lock:
                    do_reset, inbox.pending_reset = inbox.pending_reset, False
                    seed, inbox.reset_seed = inbox.reset_seed, None
                    do_resync, inbox.pending_resync = inbox.pending_resync, False
                    keys = dict(inbox.keys)
                    seq, t_client = inbox.seq, inbox.t_client
                    init_mode, gain, speed = (
                        inbox.init_mode,
                        inbox.diff_gain,
                        inbox.speed,
                    )
                    stale = (time.perf_counter() - inbox.last_msg_t) > STALE_S

                if do_reset:
                    await ws.send_json({"type": "resetting"})
                    res = await loop.run_in_executor(
                        EXEC, RUNNER.reset, seed, init_mode, gain
                    )
                    await send_tick(ws, res, seq, t_client)
                    next_t = time.perf_counter()
                    continue
                if do_resync:
                    res = await loop.run_in_executor(EXEC, RUNNER.resync, gain)
                    await send_tick(ws, res, seq, t_client)
                    continue

                delta = parse_delta(keys, RUNNER.delta_scale) * speed
                if stale or not delta.any():
                    continue  # frozen: nothing steps, nothing is sent
                try:
                    res = await loop.run_in_executor(EXEC, RUNNER.step, delta, gain)
                except torch.OutOfMemoryError:  # pragma: no cover
                    torch.cuda.empty_cache()
                    await ws.send_json({"type": "error", "code": "oom"})
                    continue
                await send_tick(ws, res, seq, t_client)
        except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError):
            pass
        finally:
            recv_task.cancel()


# ------------------------------------------------------------------- headless CLI --
def _main() -> None:
    """Drive the engine without a browser, to validate alignment and timing."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--init", default="fixed", choices=["fixed", "random", "wm_train"])
    ap.add_argument("--action-lag", type=int, default=1, choices=[0, 1])
    ap.add_argument("--dec-infer-steps", type=int, default=2)
    ap.add_argument("--settle-steps", type=int, default=100)
    ap.add_argument("--arm-spread", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--keys", default="w", help="keys held every tick, e.g. 'wl'")
    args = ap.parse_args()

    runner = LockstepRunner(
        device=args.device,
        seed=args.seed,
        init_mode=args.init,
        action_lag=args.action_lag,
        dec_infer_steps=args.dec_infer_steps,
        settle_steps=args.settle_steps,
        arm_spread=args.arm_spread,
    )
    runner.warmup()
    res = runner.reset(args.seed, args.init, 3.0)
    print(f"reset: psnr_floor={res.header['metrics']['psnr_db_floor']:.2f} dB")

    # pack/unpack round-trip check
    imgs = [(n, *encode_image(f, "png")) for n, f in res.frames]
    hdr2, blobs = unpack_frame(pack_frame(res.header, imgs))
    assert hdr2["tick"] == res.header["tick"] and set(blobs) == {"gt", "wm", "diff"}
    print(f"wire round-trip ok ({sum(len(b) for b in blobs.values()) / 1024:.1f} KB)")

    keys = {k: 1 for k in args.keys}
    delta = parse_delta(keys, runner.delta_scale)
    tot: List[float] = []
    for i in range(args.ticks):
        res = runner.step(delta, 3.0)
        m, t = res.header["metrics"], res.header["timing_ms"]
        tot.append(t["total"])
        if i < 3 or (i + 1) % 10 == 0:
            wm_a = m["angle_wm_deg"]
            wm_s = "  None" if wm_a is None else f"{wm_a:+6.1f}"
            print(
                f"  t={i + 1:3d}  psnr={m['psnr_db']:5.2f}"
                f"  mean={m['psnr_db_mean']:5.2f}"
                f"  static={m['psnr_db_static']:5.2f}"
                f"  gt={m['angle_gt_deg']:+6.1f}  wm={wm_s}"
                f"  px={m['t_pixels']:4d}  {t['total']:5.1f} ms"
            )
    print(
        f"timing: mean {np.mean(tot):.1f} ms  p90 {np.percentile(tot, 90):.1f} ms "
        f"(budget {FRAME_DT * 1e3:.0f} ms)"
    )


if __name__ == "__main__":
    _main()

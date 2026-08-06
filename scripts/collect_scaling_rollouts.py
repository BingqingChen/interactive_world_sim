"""Record rollout videos for each policy in the real-vs-imagined scaling figure.

One condition per plot point (# real demos x imagined dose). Each condition's policy
is rolled out under the *published* eval protocol -- random-init rotate-T, seed 7000,
the same reset sequence `eval_dp_rotate_t.py` uses -- so episode k here is exactly
episode k of the 50-episode eval behind `outputs/scaling_plot_final.png`, and every
condition sees the identical initial states.

Videos are written at the true 10 Hz control rate (i.e. real time), so the browser can
apply an honest speed multiplier with `video.playbackRate`.

The success rates printed here are from this short n-episode run and are NOT the
published numbers; the figure's own n=50 rates come from the sweep status logs in
`outputs/random_init_evals/`. Both land in the manifest, kept clearly separate.

Usage:
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python scripts/collect_scaling_rollouts.py \
      --n_episodes 10 --out_dir outputs/scaling_rollouts
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "data_collection"))

import collect_rotate_t as C  # noqa: E402
import eval_dp_rotate_t as E  # noqa: E402
import gym_aloha.env as gae  # noqa: E402
from gym_aloha.env import AlohaEnv  # noqa: E402
from yixuan_utilities.kinematics_helper import KinHelper  # noqa: E402

from interactive_world_sim.utils.pose_utils import PoseType, pose_convert  # noqa: E402

SCALING_ROOT = Path(
    "/home/jacobhb/projects/worth_doing/diffusion_policy/data/outputs/scaling"
)
STATUS_LOGS = [
    _REPO_ROOT / "outputs" / "random_init_evals" / "b2_status_mirror.log",
    _REPO_ROOT / "outputs" / "random_init_evals" / "v2_status.log",
]
PUBLISHED_RE = re.compile(
    r"SCALE-EVAL-DONE sc_r(\d+)i(\d+)_s(\d+): SUCCESS RATE:\s+([0-9.]+)%"
)
CONTROL_HZ = 10  # the rate the policy and sim actually run at


def published_rates() -> dict:
    """Per-condition n=50 success rates behind the figure, keyed by '<real>_<dose>'."""
    per_seed: dict = defaultdict(dict)
    for path in STATUS_LOGS:
        if not path.exists():
            continue
        for line in path.read_text(errors="ignore").splitlines():
            m = PUBLISHED_RE.search(line)
            if m:
                per_seed[(int(m[1]), int(m[2]))][int(m[3])] = float(m[4])
    out = {}
    for (real, dose), seeds in per_seed.items():
        v = np.array(list(seeds.values()))
        out[f"{real}_{dose}"] = {
            "real": real,
            "dose": dose,
            "mean": float(v.mean()),
            "sem": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
            "per_seed": {str(s): r for s, r in sorted(seeds.items())},
            "n_episodes": 50,
        }
    return out


def find_checkpoint(real: int, dose: int, prefer_seed: int) -> tuple:
    """Locate a trained checkpoint for one condition, preferring `prefer_seed`."""
    cands = sorted(SCALING_ROOT.glob(f"sc_r{real}i{dose}_s*/checkpoints/latest.ckpt"))
    if not cands:
        return None, None
    for c in cands:
        if c.parent.parent.name.endswith(f"_s{prefer_seed}"):
            return c, prefer_seed
    seed = int(cands[0].parent.parent.name.rsplit("_s", 1)[1])
    return cands[0], seed


def write_video(frames: list, path: Path, fps: int, upscale: int) -> None:
    """Write RGB frames to an H.264 mp4 at `fps` (real time when fps == CONTROL_HZ)."""
    import imageio.v2 as imageio

    up = np.stack(
        [
            cv2.resize(f, (upscale, upscale), interpolation=cv2.INTER_NEAREST)
            for f in frames
        ]
    )
    imageio.mimwrite(
        str(path),
        up,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        output_params=["-crf", "23"],
    )


@torch.no_grad()
def run_condition(
    real: int, dose: int, args: argparse.Namespace, out_root: Path
) -> dict:
    """Roll one condition's policy out for n episodes, writing a video for each."""
    ckpt, seed_used = find_checkpoint(real, dose, args.prefer_seed)
    if ckpt is None:
        print(f"  r{real} i{dose}: no local checkpoint, skipping", flush=True)
        return {}

    device = torch.device(args.device)
    policy, n_obs, n_act = E.load_policy(str(ckpt), device)
    kin = KinHelper(robot_name="trossen_vx300s")
    env = AlohaEnv("pusht")
    # Published protocol: random T XY within +-0.08, upright, plus the arm settle.
    gae.sample_pusht_pose = C.make_upright_pose_sampler((-0.08, 0.08), (-0.08, 0.08))
    contact_sets = E.build_contact_sets(env)

    cond_dir = out_root / f"r{real}i{dose}"
    cond_dir.mkdir(parents=True, exist_ok=True)

    episodes, trial = [], 0
    for ep in range(args.n_episodes):
        # Identical to eval_dp_rotate_t.main: this makes episode k the same initial
        # state for every condition, and the same as the published eval's episode k.
        np.random.seed(args.seed + ep)
        while True:
            env.reset(seed=args.seed + trial)
            trial += 1
            C.stabilize_t(env)
            o = env._env.task.get_observation(env._env.physics)  # noqa: SLF001
            bases = np.stack(
                [
                    pose_convert(o["left_base"][None], PoseType.POS_QUAT, PoseType.MAT)[
                        0
                    ],
                    pose_convert(
                        o["right_base"][None], PoseType.POS_QUAT, PoseType.MAT
                    )[0],
                ]
            )
            C.settle_arms(
                env, bases, kin, np.zeros(6), E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM
            )
            if abs(C.t_angle(env)) <= C.UPRIGHT_TOL:
                break

        t0 = time.time()
        init_pose, final_pose, steps, frames, contacts = E.run_episode(
            env,
            policy,
            kin,
            bases,
            n_obs,
            n_act,
            args.max_steps,
            True,
            contact_sets=contact_sets,
        )
        ok, delta = E.eval_success(init_pose, final_pose)
        deg = float(np.degrees(delta))
        name = f"ep{ep:02d}_{'ok' if ok else 'fail'}_{deg:+.0f}deg.mp4"
        write_video(frames, cond_dir / name, args.fps, args.upscale)
        episodes.append(
            {
                "episode": ep,
                "video": f"r{real}i{dose}/{name}",
                "success": bool(ok),
                "rotation_deg": round(deg, 1),
                "steps": int(steps),
                "contact_events": int(contacts["contact_events"]),
                "duration_s": round(steps / CONTROL_HZ, 1),
            }
        )
        print(
            f"  r{real} i{dose} ep{ep:02d}: {deg:+6.1f} deg  steps={steps:3d}  "
            f"{'SUCCESS' if ok else 'fail   '}  ({time.time() - t0:.0f}s)",
            flush=True,
        )

    n_ok = sum(e["success"] for e in episodes)
    return {
        "real": real,
        "dose": dose,
        "checkpoint": str(ckpt),
        "seed_used": seed_used,
        "n_episodes": len(episodes),
        "sampled_success_rate": 100.0 * n_ok / max(len(episodes), 1),
        "sampled_rotation_deg_mean": round(
            float(np.mean([e["rotation_deg"] for e in episodes])), 1
        ),
        "episodes": episodes,
    }


def main() -> None:
    """Record rollouts for every scaling condition and write a manifest."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="outputs/scaling_rollouts")
    ap.add_argument("--n_episodes", type=int, default=10)
    # 800, not eval_dp_rotate_t.py's 300 default: every one of the 45 published runs
    # used 800, and 65% of their successful episodes take more than 300 steps. Capping
    # at 300 truncates most successes mid-attempt and makes the videos disagree with
    # the plotted rates.
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=7000, help="published eval seed")
    ap.add_argument("--prefer_seed", type=int, default=3, help="training seed to show")
    ap.add_argument("--fps", type=int, default=CONTROL_HZ, help="10 = real time")
    # 128 = the frames' native size. Upscaling here only inflates the file (a 256px
    # nearest-neighbour copy is ~2.7x the bytes for zero extra information, and the
    # hard edges it creates actually hurt compression). The page scales up in CSS with
    # image-rendering: pixelated, which looks identical.
    ap.add_argument("--upscale", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_root = _REPO_ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    pub = published_rates()
    print(f"{len(pub)} conditions with published rates\n")

    conditions = sorted(
        ((v["real"], v["dose"]) for v in pub.values()), key=lambda t: (t[0], t[1])
    )
    manifest = {
        "protocol": {
            "eval_seed": args.seed,
            "max_steps": args.max_steps,
            "n_episodes": args.n_episodes,
            "fps": args.fps,
            "control_hz": CONTROL_HZ,
            "init": "random T XY +-0.08 upright + arm settle",
            "note": (
                "Videos are an n={n} sample recorded here. The plotted success rates "
                "are the published n=50 numbers from outputs/random_init_evals/."
            ).format(n=args.n_episodes),
        },
        "published": pub,
        "conditions": {},
    }
    manifest_path = out_root / "manifest.json"

    t_start = time.time()
    for k, (real, dose) in enumerate(conditions, 1):
        print(f"[{k}/{len(conditions)}] r{real} i{dose}", flush=True)
        res = run_condition(real, dose, args, out_root)
        if res:
            manifest["conditions"][f"{real}_{dose}"] = res
            # Written incrementally so the page is usable while the sweep runs.
            manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"\nwrote {manifest_path} "
        f"({len(manifest['conditions'])} conditions, {time.time() - t_start:.0f}s)"
    )


if __name__ == "__main__":
    main()

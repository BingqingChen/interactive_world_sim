"""Convert openpi robosuite-Lift pi0.5 rollouts (per-episode .npz) to IWS episode-per-file HDF5.

The npz rollouts are produced by openpi's examples/libero/main_lift.py. This writes the SAME
episode format as convert_robomimic_to_hdf5.py, so the output drops straight into SimAlohaDataset
via configurations/dataset/sim_robosuite_dataset.yaml (dataset_dir: data/robosuite/lift).

Field provenance (npz -> IWS episode):
    action                 <- npz["actions"]  (T,7)  [only its length T is used by the loader]
    timestamp              <- arange(T)/20    (robosuite control_freq)
    obs/joint_pos  (T,7)   <- npz["sim_state"][:, 1:8]   (qpos[0:7] = the 7 Panda arm joints;
                                                          verified bit-exact vs the robosuite API)
    obs/ee_pos     (T,1,4,4) <- SE(3) from npz["state"][:, :3] (eef_pos) + [:, 3:6] (axis-angle)
    obs/world_t_robot_base (T,1,4,4) identity (fixed-base Panda)
    obs/images/camera_0_color <- resize(npz["agentview"], 128, INTER_AREA)   [agentview]
    obs/images/camera_1_color <- resize(npz["wrist"],     128, INTER_AREA)   [robot0_eye_in_hand]

ALL 1000 episodes (successes AND failures) convert identically -- the format carries no
success/failure field; goal_sample:intermediate makes every episode a trajectory whose goal is a
later frame from the same episode. Failures simply add trajectories where the cube is never lifted.

Usage (IWS venv):
    .venv/bin/python scripts/data_collection/convert_openpi_lift_npz_to_hdf5.py \\
        -i /home/jacobhb/projects/openpi/data/robosuite/rollouts_lift_pi \\
        -o data/robosuite/lift
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List

import click
import cv2
import numpy as np
from tqdm import tqdm

try:
    from yixuan_utilities.hdf5_utils import save_dict_to_hdf5
except ImportError:
    from interactive_world_sim.algorithms.common.hdf5_utils import save_dict_to_hdf5

ROBOSUITE_CONTROL_FREQ = 20  # Hz


def _axisangle_to_mat(axisangle: np.ndarray) -> np.ndarray:
    """(T, 3) rotation vectors (axis * angle) -> (T, 3, 3) rotation matrices via Rodrigues."""
    return np.stack([cv2.Rodrigues(v.astype(np.float64))[0] for v in axisangle], axis=0)


def _resize_images(imgs: np.ndarray, resolution: int) -> np.ndarray:
    """INTER_AREA resize each frame in (T, H, W, 3) to (T, resolution, resolution, 3)."""
    if imgs.shape[1] == resolution and imgs.shape[2] == resolution:
        return imgs
    return np.stack(
        [cv2.resize(f, (resolution, resolution), interpolation=cv2.INTER_AREA) for f in imgs],
        axis=0,
    )


def build_episode(d: "np.lib.npyio.NpzFile", resolution: int) -> Dict:
    """Read one openpi Lift npz and return an IWS-format episode dict."""
    actions = np.asarray(d["actions"], dtype=np.float32)  # (T, 7): 6 OSC motion dims + gripper cmd
    T = actions.shape[0]
    state = np.asarray(d["state"], dtype=np.float32)  # (T, 8): eef_pos(3)+axisangle(3)+grip(2)
    sim_state = np.asarray(d["sim_state"])  # (T, 32): [time, qpos(16), qvel(15)]

    # joint_pos: the 7 Panda arm joints live at qpos[0:7] = sim_state[:, 1:8].
    joint_pos = sim_state[:, 1:8].astype(np.float32)  # (T, 7)

    # gripper COMMAND: the policy's 7th action dim (-1 open / +1 close), clipped to the range the
    # OSC controller actually applies. This is the control signal the world model conditions on
    # (action_mode "single_grasp_cmd"); joint_pos[:,-1] (wrist) carries no grasp information.
    gripper_cmd = np.clip(actions[:, 6:7], -1.0, 1.0).astype(np.float32)  # (T, 1)

    # ee_pos: SE(3) from recorded eef translation + axis-angle rotation.
    rot = _axisangle_to_mat(state[:, 3:6]).astype(np.float32)  # (T, 3, 3)
    ee_mat = np.broadcast_to(np.eye(4, dtype=np.float32), (T, 4, 4)).copy()
    ee_mat[:, :3, :3] = rot
    ee_mat[:, :3, 3] = state[:, :3]
    ee_pos = ee_mat[:, None, :, :]  # (T, 1, 4, 4)

    world_t_robot_base = np.broadcast_to(
        np.eye(4, dtype=np.float32)[None, None], (T, 1, 4, 4)
    ).copy()

    cam0 = _resize_images(np.asarray(d["agentview"], dtype=np.uint8), resolution)  # agentview
    cam1 = _resize_images(np.asarray(d["wrist"], dtype=np.uint8), resolution)  # wrist

    timestamp = (np.arange(T) / ROBOSUITE_CONTROL_FREQ).astype(np.float64)

    return {
        "action": actions,
        "timestamp": timestamp,
        "obs": {
            "joint_pos": joint_pos,
            "gripper": gripper_cmd,
            "ee_pos": ee_pos,
            "world_t_robot_base": world_t_robot_base,
            "images": {
                "camera_0_color": cam0,
                "camera_1_color": cam1,
            },
        },
    }


def _verify_episode(episode: Dict, resolution: int) -> None:
    T = episode["action"].shape[0]
    assert episode["action"].dtype == np.float32
    assert episode["timestamp"].shape == (T,)
    assert episode["obs"]["joint_pos"].shape == (T, 7)
    assert episode["obs"]["gripper"].shape == (T, 1)
    assert episode["obs"]["ee_pos"].shape == (T, 1, 4, 4)
    assert episode["obs"]["world_t_robot_base"].shape == (T, 1, 4, 4)
    for cam_key in ("camera_0_color", "camera_1_color"):
        img = episode["obs"]["images"][cam_key]
        assert img.shape == (T, resolution, resolution, 3), f"{cam_key}: {img.shape}"
        assert img.dtype == np.uint8


def _save_episode(episode: Dict, split_dir: str, episode_id: int, attr_dict: Dict, resolution: int) -> None:
    H = W = resolution
    config_dict: Dict = {
        "timestamp": {"dtype": "float64"},
        "obs": {
            "images": {
                "camera_0_color": {"chunks": (1, H, W, 3), "dtype": "uint8"},
                "camera_1_color": {"chunks": (1, H, W, 3), "dtype": "uint8"},
            }
        },
    }
    episode_path = os.path.join(split_dir, f"episode_{episode_id}.hdf5")
    save_dict_to_hdf5(episode, config_dict, episode_path, attr_dict=attr_dict)


@click.command()
@click.option("--input", "-i", "input_dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="Directory of openpi Lift npz rollouts (searched recursively for episode_*.npz).")
@click.option("--output_dir", "-o", required=True, type=click.Path(),
              help="Root output dir; train/ and val/ are created here.")
@click.option("--val_ratio", default=0.1, show_default=True, type=float)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--resolution", default=128, show_default=True, type=int)
@click.option("--limit", default=-1, show_default=True, type=int,
              help="Convert only the first N episodes (smoke test). -1 = all.")
def main(input_dir: str, output_dir: str, val_ratio: float, seed: int, resolution: int, limit: int) -> None:
    paths: List[str] = sorted(
        glob.glob(os.path.join(input_dir, "**", "episode_*.npz"), recursive=True)
    )
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"No episode_*.npz found under {input_dir}")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(paths))
    n_val = max(1, int(len(paths) * val_ratio))
    val_set = set(perm[:n_val].tolist())

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)

    train_id = val_id = 0
    n_success = 0
    for i, p in enumerate(tqdm(paths, unit="ep")):
        d = np.load(p)
        success = bool(d["success"])
        n_success += int(success)
        attr_dict = {
            "sim": True,
            "source": "openpi_lift_pi",
            "source_file": os.path.basename(p),
            "env_name": str(d["env_name"]),
            "success": success,
            "prompt": str(d["prompt"]),
        }
        episode = build_episode(d, resolution)
        _verify_episode(episode, resolution)
        if i in val_set:
            _save_episode(episode, val_dir, val_id, attr_dict, resolution)
            val_id += 1
        else:
            _save_episode(episode, train_dir, train_id, attr_dict, resolution)
            train_id += 1

    print(
        f"\nDone. {len(paths)} episodes ({n_success} success / {len(paths) - n_success} failure) "
        f"-> {train_id} train / {val_id} val in {output_dir} @ {resolution}px"
    )


if __name__ == "__main__":
    main()

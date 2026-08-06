"""Scripted collection of "reach and TOUCH the T" demonstrations (no rotation).

Same random-init protocol as collect_rotate_t.py — the T spawns UPRIGHT at a
random XY (patched `sample_pusht_pose`), falls and settles (`stabilize_t`), and
the arms settle to a random ready pose (`settle_arms`, skippable) — but the task
is different: BOTH arms only reach toward the block and make gripper contact
with it, then stop. No twisting, no pushing.

Each arm heads for the same validated side grip points the rotation task uses
(upright-T branch), via an approach point with a small random angle offset, plus
a slight inward nudge so fingertip contact is guaranteed under PID lag. The
episode ends HOLD_STEPS after both arms have registered contact (MuJoCo
gripper<->T contact check each 10 Hz control step).

Accepted demos require: both arms touched, the T stayed flat and essentially
unmoved (|rotation| < 10 deg, |XY displacement| < 2 cm), actions in range.
Output HDF5/mp4 layout is identical to the other collectors (action (T,4)
bimanual EE-XY, obs/images/top_pov 128x128, videos/episode_N.mp4).

Usage:
    MUJOCO_GL=egl python scripts/data_collection/collect_touch_t.py \
        --output_dir datasets/touch_t --n_episodes 1 --headless
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import time
from pathlib import Path

import click
import cv2
import numpy as np

import gym_aloha.env as gae
from gym_aloha.env import AlohaEnv
from yixuan_utilities.kinematics_helper import KinHelper

sys.path.insert(0, str(Path(__file__).parent))
from sim_aloha_dataset_collection_scripted import (  # noqa: E402
    env_state_to_mat,
    extract_t_info,
    get_current_arm_positions,
    init_episode,
    save_episode,
    trajectory_to_joint_actions,
    update_episode,
    vis_obs,
)

import collect_rotate_t as C  # noqa: E402  (spawn sampler, stabilize_t, settle_arms)
from interactive_world_sim.utils.mujoco_contacts import (  # noqa: E402
    build_contact_sets,
    gripper_t_contact,
)
from interactive_world_sim.utils.motion_planner import (  # noqa: E402
    TGeometryAnalyzer,
    actions_in_range,
)
from interactive_world_sim.utils.pose_utils import PoseType, pose_convert  # noqa: E402
from interactive_world_sim.utils.trajectory_primitives import (  # noqa: E402
    BimanualCoordination,
    CurvePrimitive,
    TrajectoryConfig,
)

REACH_NUM_STEPS = 80  # control steps planned for the reach (episode usually ends earlier)
NUDGE = 0.015  # m past the grip point toward the T center, guarantees fingertip contact
HOLD_STEPS = 10  # extra recorded steps after both arms are touching (visible dwell)
MAX_ROT_DEG = 10.0  # accepted demo: T rotated less than this
MAX_DISP = 0.02  # accepted demo: T center moved less than this (m)


def plan_touch(t_analyzer: TGeometryAnalyzer, current_arm_xy, config, coordinator):
    """Approach -> side grip point -> small inward nudge, for both arms.
    Grip selection matches the upright-T branch of the rotation planner."""
    kp = t_analyzer.key_points
    center = t_analyzer.t_info.center
    left_grip = kp["top_bar_bottom_side_left"]
    right_grip = kp["top_right"]

    la = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
    ra = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
    left_approach = left_grip - np.array([np.sin(la), np.cos(la)]) * C.APPROACH_DISTANCE
    right_approach = right_grip + np.array([np.sin(ra), np.cos(ra)]) * C.APPROACH_DISTANCE

    def nudge(p):
        d = center - p
        n = np.linalg.norm(d)
        return p + (d / n) * NUDGE if n > 1e-6 else p

    left_wp = [left_approach, left_grip, nudge(left_grip)]
    right_wp = [right_approach, right_grip, nudge(right_grip)]
    for p in left_wp:
        p[0] -= C.EEF_FINGERTIP_OFFSET
    for p in right_wp:
        p[0] += C.EEF_FINGERTIP_OFFSET
    left_wp = [current_arm_xy[:2], *left_wp]
    right_wp = [current_arm_xy[2:], *right_wp]

    return coordinator.coordinate(
        CurvePrimitive(left_wp, config),
        CurvePrimitive(right_wp, config),
        duration=REACH_NUM_STEPS / 10.0,
        num_steps=REACH_NUM_STEPS,
        sync_type="simultaneous",
        speed_profile="constant",
    )


def touch_success(init_pose, final_pose, actions, left_touched, right_touched):
    """Both arms touched; T flat, unrotated (<MAX_ROT_DEG) and unmoved (<MAX_DISP)."""
    table_n = np.array([0, 0, 1])
    flat = (init_pose[:3, 2] @ table_n > 0.95) and (final_pose[:3, 2] @ table_n > 0.95)
    d = np.linalg.inv(init_pose) @ final_pose
    rot = abs(np.degrees(np.arctan2(d[1, 0], d[0, 0])))
    disp = float(np.linalg.norm(final_pose[:2, 3] - init_pose[:2, 3]))
    ok = (left_touched and right_touched and flat and rot < MAX_ROT_DEG
          and disp < MAX_DISP and actions_in_range(actions))
    return ok, rot, disp


@click.command()
@click.option("--output_dir", "-o", default="datasets/touch_t")
@click.option("--n_episodes", "-n", default=100, type=int)
@click.option("--headless", "-h", is_flag=True)
@click.option("--seed", default=0, type=int)
@click.option("--x_min", default=-0.08, type=float)
@click.option("--x_max", default=0.08, type=float)
@click.option("--y_min", default=-0.08, type=float)
@click.option("--y_max", default=0.08, type=float)
@click.option("--no_settle", is_flag=True,
              help="Skip the random arm-settle step (fixed home arm start).")
def main(output_dir, n_episodes, headless, seed, x_min, x_max, y_min, y_max, no_settle):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    gae.sample_pusht_pose = C.make_upright_pose_sampler((x_min, x_max), (y_min, y_max))

    kin_helper = KinHelper(robot_name="trossen_vx300s")
    env = AlohaEnv("pusht")
    cv2.setNumThreads(1)
    contact_sets = build_contact_sets(env)

    dt = 1 / 10.0
    k_p, k_v = 50, 10
    acc_lim, vel_lim = 10.0, 0.04
    config = TrajectoryConfig(noise_std=0.001)
    coordinator = BimanualCoordination(config)

    episode_id = len(list(Path(output_dir).glob("episode_*.hdf5")))
    init_episode_id = episode_id
    trial = 0
    target_total = episode_id + n_episodes

    print(f"Collecting {n_episodes} touch-T demos into {output_dir} "
          f"(starting at episode_{episode_id})")

    while episode_id < target_total:
        ep_seed = seed * 1_000_000 + trial
        trial += 1
        env.reset(seed=ep_seed)
        C.stabilize_t(env)  # let the T fall/settle before observing anything

        obs = env._env.task.get_observation(env._env.physics)
        lb = pose_convert(obs["left_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
        rb = pose_convert(obs["right_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
        world_t_bases = np.stack([lb, rb])

        curr_vel = np.zeros(6)
        if not no_settle:
            C.settle_arms(env, world_t_bases, kin_helper, curr_vel, dt, k_p, k_v,
                          acc_lim, vel_lim)
        curr_vel[:] = 0.0
        if abs(C.t_angle(env)) > C.UPRIGHT_TOL:
            print(f"  [skip] trial {trial}: settle bumped the T -- discarded")
            continue

        obs = env._env.task.get_observation(env._env.physics)
        t_info = extract_t_info(obs["env_state"])
        current_arm_xy = get_current_arm_positions(obs, kin_helper, world_t_bases)
        traj = plan_touch(TGeometryAnalyzer(t_info), current_arm_xy, config, coordinator)

        episode = init_episode()
        left_touched = right_touched = False
        hold = 0
        t0 = time.time()
        for target_xy in traj:
            obs = env._env.task.get_observation(env._env.physics)
            joint, target_xy_clip = trajectory_to_joint_actions(
                target_xy, world_t_bases, kin_helper, obs["qpos"][:14],
                curr_vel, dt, k_p, k_v, acc_lim, vel_lim,
            )
            episode = update_episode(episode, obs, target_xy_clip, kin_helper)
            if not headless:
                vis_img = vis_obs(obs, episode_id, True, env)
                cv2.imshow("Touch-T Collection", cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            env.step(joint)
            in_l, in_r = gripper_t_contact(env, contact_sets)
            left_touched |= in_l
            right_touched |= in_r
            if left_touched and right_touched:
                hold += 1
                if hold >= HOLD_STEPS:
                    break

        if len(episode["action"]) == 0:
            continue
        init_pose = env_state_to_mat(episode["env_state"][0])
        final_pose = env_state_to_mat(episode["env_state"][-1])
        actions = np.stack(episode["action"])
        ok, rot, disp = touch_success(init_pose, final_pose, actions,
                                      left_touched, right_touched)
        if ok:
            save_episode(episode, output_dir, episode_id)
            episode_id += 1
            print(f"  [OK]   episode_{episode_id - 1}: touched (L={left_touched} "
                  f"R={right_touched}), T moved {disp*100:.1f} cm / {rot:.1f} deg, "
                  f"{len(actions)} frames ({time.time()-t0:.0f}s) "
                  f"({episode_id - init_episode_id}/{n_episodes})")
        else:
            print(f"  [fail] trial {trial}: L={left_touched} R={right_touched} "
                  f"moved {disp*100:.1f} cm / rot {rot:.1f} deg -- discarded")

    print(f"Done. {episode_id - init_episode_id} demos saved to {output_dir}, "
          f"success rate {(episode_id - init_episode_id) / max(trial, 1):.2f}")


if __name__ == "__main__":
    main()

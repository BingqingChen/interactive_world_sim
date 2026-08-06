"""Scripted collection of arms-only demos: NO T block in view, just arm motion.

Control condition for the touch-T data (collect_touch_t.py): the identical
random-init protocol and the identical reach-style trajectories, but the T is
spawned far outside the camera view, so the scene contains only the table and
the two arms. Each episode plans the same approach->grip->nudge reach toward a
VIRTUAL upright T pose (sampled with the usual random-XY sampler) and executes
a random 50-60 control steps of it (5-6 s at 10 Hz) — matching the touch demos'
motion profile without any object contact.

Output HDF5/mp4 layout is identical to the other collectors. (env_state records
the off-screen T pose; downstream DP conversion only uses img/ee_pos/action.)

Usage:
    MUJOCO_GL=egl python scripts/data_collection/collect_no_t.py \
        --output_dir datasets/no_t --n_episodes 100 --headless
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
    extract_t_info,
    get_current_arm_positions,
    init_episode,
    save_episode,
    trajectory_to_joint_actions,
    update_episode,
    vis_obs,
)

import collect_rotate_t as C  # noqa: E402  (upright sampler, stabilize_t, settle_arms)
from collect_touch_t import plan_touch  # noqa: E402  (same reach trajectories)
from interactive_world_sim.utils.motion_planner import (  # noqa: E402
    TGeometryAnalyzer,
    actions_in_range,
)
from interactive_world_sim.utils.pose_utils import PoseType, pose_convert  # noqa: E402
from interactive_world_sim.utils.trajectory_primitives import (  # noqa: E402
    BimanualCoordination,
    TrajectoryConfig,
)

OFFSCREEN_POSE = np.array([1.5, 0.0, 0.07, 1.0, 0.0, 0.0, 0.0])  # far off-camera
MIN_STEPS, MAX_STEPS = 50, 60  # 5-6 s at 10 Hz


@click.command()
@click.option("--output_dir", "-o", default="datasets/no_t")
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

    # Spawn the REAL T far outside the top_pov view: the scene shows only arms.
    gae.sample_pusht_pose = lambda seed=None: OFFSCREEN_POSE.copy()
    # Virtual T poses (never instantiated) drive the reach trajectories.
    virtual_pose_sampler = C.make_upright_pose_sampler((x_min, x_max), (y_min, y_max))

    kin_helper = KinHelper(robot_name="trossen_vx300s")
    env = AlohaEnv("pusht")
    cv2.setNumThreads(1)

    dt = 1 / 10.0
    k_p, k_v = 50, 10
    acc_lim, vel_lim = 10.0, 0.04
    config = TrajectoryConfig(noise_std=0.001)
    coordinator = BimanualCoordination(config)

    episode_id = len(list(Path(output_dir).glob("episode_*.hdf5")))
    init_episode_id = episode_id
    trial = 0
    target_total = episode_id + n_episodes

    print(f"Collecting {n_episodes} arms-only (no T) demos into {output_dir} "
          f"(starting at episode_{episode_id})")

    while episode_id < target_total:
        ep_seed = seed * 1_000_000 + trial
        trial += 1
        env.reset(seed=ep_seed)
        C.stabilize_t(env)  # protocol parity (the T settles off-screen)

        obs = env._env.task.get_observation(env._env.physics)
        lb = pose_convert(obs["left_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
        rb = pose_convert(obs["right_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
        world_t_bases = np.stack([lb, rb])

        curr_vel = np.zeros(6)
        if not no_settle:
            C.settle_arms(env, world_t_bases, kin_helper, curr_vel, dt, k_p, k_v,
                          acc_lim, vel_lim)
        curr_vel[:] = 0.0

        # Same reach trajectory as touch demos, toward a virtual T.
        rng = np.random.RandomState(ep_seed)
        virtual_state = virtual_pose_sampler(seed=ep_seed)
        t_info = extract_t_info(virtual_state)
        obs = env._env.task.get_observation(env._env.physics)
        current_arm_xy = get_current_arm_positions(obs, kin_helper, world_t_bases)
        traj = plan_touch(TGeometryAnalyzer(t_info), current_arm_xy, config, coordinator)
        n_steps = int(rng.randint(MIN_STEPS, MAX_STEPS + 1))

        episode = init_episode()
        t0 = time.time()
        for target_xy in traj[:n_steps]:
            obs = env._env.task.get_observation(env._env.physics)
            joint, target_xy_clip = trajectory_to_joint_actions(
                target_xy, world_t_bases, kin_helper, obs["qpos"][:14],
                curr_vel, dt, k_p, k_v, acc_lim, vel_lim,
            )
            episode = update_episode(episode, obs, target_xy_clip, kin_helper)
            if not headless:
                vis_img = vis_obs(obs, episode_id, True, env)
                cv2.imshow("No-T Collection", cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            env.step(joint)

        actions = np.stack(episode["action"])
        if actions_in_range(actions):
            save_episode(episode, output_dir, episode_id)
            episode_id += 1
            print(f"  [OK]   episode_{episode_id - 1}: {len(actions)} frames "
                  f"({time.time()-t0:.0f}s) ({episode_id - init_episode_id}/{n_episodes})")
        else:
            print(f"  [fail] trial {trial}: actions out of range -- discarded")

    print(f"Done. {episode_id - init_episode_id} demos saved to {output_dir}, "
          f"success rate {(episode_id - init_episode_id) / max(trial, 1):.2f}")


if __name__ == "__main__":
    main()

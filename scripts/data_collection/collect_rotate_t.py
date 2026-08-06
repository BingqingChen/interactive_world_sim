"""Scripted collection of "rotate the T 90 degrees to the right" demonstrations.

Reuses the same MuJoCo/ALOHA machinery that produced the world-model data
(scripts/data_collection/sim_aloha_dataset_collection_scripted.py), but with two
task-specific changes:

  1. Initial state: the T is spawned UPRIGHT (top bar parallel to the table's top
     edge, i.e. z-rotation theta = 0) at a random XY, instead of a uniformly random
     orientation. This is enforced by patching gym_aloha's `sample_pusht_pose`.

  2. Motion: a CLOSED-LOOP clockwise ("to the right") rotation. Because two-finger
     pushing slips, we don't command one open-loop 90 deg swing. Instead we re-read
     the T's true angle after each ~30 deg clockwise sub-rotation (reusing the
     validated grip selection from the built-in rotating motion, but sweeping about
     the T's own center) and keep issuing sub-rotations until the T has ACTUALLY
     turned 90 deg. Slippage is absorbed by the feedback loop.

Output HDF5/mp4 layout is identical to the world-model dataset (action (T,4) bimanual
EE-XY, obs/images/top_pov 128x128, joint_pos, ee_pos), so it is drop-in for BC.

Usage:
    MUJOCO_GL=egl python scripts/data_collection/collect_rotate_t.py \
        --output_dir datasets/rotate_t --n_episodes 100 --headless
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import time
from pathlib import Path

import click
import cv2
import numpy as np
import transforms3d

import gym_aloha.env as gae
from gym_aloha.env import AlohaEnv
from yixuan_utilities.kinematics_helper import KinHelper

# Reuse the proven helpers from the world-model data collector.
sys.path.insert(0, str(Path(__file__).parent))
from sim_aloha_dataset_collection_scripted import (  # noqa: E402
    env_state_to_mat,
    extract_t_info,
    generate_random_init_action,
    get_current_arm_positions,
    init_episode,
    save_episode,
    trajectory_to_joint_actions,
    update_episode,
    vis_obs,
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

# ----------------------------------------------------------------------------- #
# Task constants
# ----------------------------------------------------------------------------- #
TARGET_ANGLE = -np.pi / 2  # 90 deg clockwise (to the right); theta 0 -> -pi/2
MAX_SUBSTEP = np.pi / 6  # cap a single sub-rotation at 30 deg (the validated size)
ANGLE_TOL = 0.12  # stop once within ~7 deg of the target (lands near a full 90)
# Accept only demos that genuinely turned ~90 deg: between 80 and 105 deg clockwise.
ACCEPT_MIN = np.radians(80.0)
ACCEPT_MAX = np.radians(105.0)
MAX_SUBROTATIONS = 10  # let slow (slipping) episodes finish the turn before giving up
MIN_PROGRESS = 0.14  # abort a trial that hasn't turned ~8 deg CW after 2 sub-rotations
UPRIGHT_TOL = np.radians(3.0)  # discard a trial if the settle bumped the T off upright
SUBSTEP_NUM_STEPS = 60  # control steps executed per sub-rotation
APPROACH_DISTANCE = 0.06
EEF_FINGERTIP_OFFSET = 0.08
N_ARC = 4  # intermediate arc waypoints per sub-rotation (smooth spline)


# ----------------------------------------------------------------------------- #
# Upright + random-XY initial pose
# ----------------------------------------------------------------------------- #
def make_upright_pose_sampler(x_range, y_range):
    """Return a drop-in replacement for gym_aloha.utils.sample_pusht_pose that
    spawns the T upright (theta = 0) at a random XY within the given ranges."""

    def sampler(seed=None):
        rng = np.random.RandomState(seed)
        x = rng.uniform(*x_range)
        y = rng.uniform(*y_range)
        theta = 0.0  # upright: top bar parallel to the table's top edge
        mat = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        quat = transforms3d.quaternions.mat2quat(mat)
        return np.concatenate([[x, y, 0.07], quat])

    return sampler


def hide_t(env) -> None:
    """Make the T block invisible to the RENDERER only (alpha = 0 on all geoms
    of body 'box'). Physics — collisions, pushing, rotation — is unaffected;
    the cameras just don't draw it. Re-apply after every env.reset()."""
    model = env._env.physics.model
    for g in range(model.ngeom):
        body = model.id2name(model.geom_bodyid[g], "body") or ""
        if body == "box":
            model.geom_rgba[g, 3] = 0.0


def stabilize_t(env, n=20) -> None:
    """Hold the arms at their reset pose for n steps so the T (spawned at z=0.07)
    falls ~5cm and settles on the table BEFORE anything is recorded or observed.
    Mirrors the 100-step init of the original world-model collection (task_reset);
    measured: the T reaches its resting height (z~0.017) within ~10 steps, with
    orientation and XY unchanged. Always run this right after env.reset()."""
    qpos0 = env._env.task.get_observation(env._env.physics)["qpos"][:14].copy()
    for _ in range(n):
        env.step(qpos0)


def t_angle(env) -> float:
    """Current z-rotation of the T in the world frame (radians)."""
    state = env._env.task.get_observation(env._env.physics)["env_state"]
    rot = transforms3d.quaternions.quat2mat(state[3:])
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


# ----------------------------------------------------------------------------- #
# One clockwise sub-rotation trajectory (grips selected as in the built-in
# rotating motion, but the arc sweeps about the T's own center and is densified).
# ----------------------------------------------------------------------------- #
def plan_cw_subrotation(
    t_analyzer: TGeometryAnalyzer,
    current_arm_xy: np.ndarray,
    step_angle: float,
    config: TrajectoryConfig,
    coordinator: BimanualCoordination,
) -> np.ndarray:
    angle = t_analyzer.t_info.rotation
    kp = t_analyzer.key_points

    # Grip selection (clockwise) copied from motion_planner.get_rotation_waypoints.
    if -np.pi / 4 < angle < np.pi / 4:  # T pointing down (upright)
        left_grip = kp["top_bar_bottom_side_left"]
        right_grip = kp["top_right"]
    elif np.pi / 4 < angle < 3 * np.pi / 4:  # pointing right
        left_grip = kp["top_left"]
        right_grip = kp["stem_bottom_right"]
    elif (3 * np.pi / 4 < angle < np.pi) or (-np.pi < angle < -3 * np.pi / 4):  # up
        left_grip = kp["top_right"]
        right_grip = kp["top_bar_bottom_side_left"]
    else:  # pointing left, angle in (-3pi/4, -pi/4)
        left_grip = kp["stem_bottom_right"]
        right_grip = kp["top_left"]

    # Approach points (small random offset, as in the built-in motion).
    la = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
    ra = np.random.uniform(-np.pi / 6.0, np.pi / 6.0)
    left_approach = left_grip - np.array([np.sin(la), np.cos(la)]) * APPROACH_DISTANCE
    right_approach = right_grip + np.array([np.sin(ra), np.cos(ra)]) * APPROACH_DISTANCE

    # Sweep the two grips clockwise about the T's CENTER (rotate in place).
    center = t_analyzer.t_info.center
    left_rel = left_grip - center
    right_rel = right_grip - center
    left_arc, right_arc = [], []
    for k in range(1, N_ARC + 1):
        phi = step_angle * k / N_ARC
        cos_r, sin_r = np.cos(phi), np.sin(phi)
        r_cw = np.array([[cos_r, sin_r], [-sin_r, cos_r]])  # clockwise
        left_arc.append(center + r_cw @ left_rel)
        right_arc.append(center + r_cw @ right_rel)

    left_wp = [left_approach, left_grip, *left_arc]
    right_wp = [right_approach, right_grip, *right_arc]

    # eef fingertip offset along x (left arm approaches from -x, right from +x).
    for p in left_wp:
        p[0] -= EEF_FINGERTIP_OFFSET
    for p in right_wp:
        p[0] += EEF_FINGERTIP_OFFSET

    left_wp = [current_arm_xy[:2], *left_wp]
    right_wp = [current_arm_xy[2:], *right_wp]

    left_primitive = CurvePrimitive(left_wp, config)
    right_primitive = CurvePrimitive(right_wp, config)
    return coordinator.coordinate(
        left_primitive,
        right_primitive,
        duration=SUBSTEP_NUM_STEPS / 10.0,
        num_steps=SUBSTEP_NUM_STEPS,
        sync_type="simultaneous",
        speed_profile="constant",
    )


def execute_trajectory(
    env,
    trajectory,
    episode,
    world_t_bases,
    kin_helper,
    curr_vel,
    dt,
    k_p,
    k_v,
    acc_lim,
    vel_lim,
    record,
    headless,
    episode_id,
):
    """Run a planned bimanual EE-XY trajectory under the PID/IK controller,
    recording (obs, action) pairs into `episode` when `record` is True."""
    for target_xy in trajectory:
        obs = env._env.task.get_observation(env._env.physics)
        curr_puppet_joint = obs["qpos"][:14]
        puppet_target_state, target_xy_clip = trajectory_to_joint_actions(
            target_xy,
            world_t_bases,
            kin_helper,
            curr_puppet_joint,
            curr_vel,
            dt,
            k_p,
            k_v,
            acc_lim,
            vel_lim,
        )
        if record:
            episode = update_episode(episode, obs, target_xy_clip, kin_helper)
        if not headless:
            vis_img = vis_obs(obs, episode_id, record, env)
            cv2.imshow("Rotate-T Collection", cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        env.step(puppet_target_state)
    return episode


def settle_arms(
    env, world_t_bases, kin_helper, curr_vel, dt, k_p, k_v, acc_lim, vel_lim, n=45
):
    """Move the arms from their fixed reset pose to a random *nearby* ready position,
    so the gripper start varies across demos. The offset is small and biased laterally
    outward (away from the T at the center) so the T is rarely bumped; any residual
    bump is caught by the uprightness check in the main loop."""
    obs = env._env.task.get_observation(env._env.physics)
    home = get_current_arm_positions(obs, kin_helper, world_t_bases)  # (4,) L_xy, R_xy
    offset = np.array(
        [
            np.random.uniform(-0.06, 0.01),  # left x  (mostly outward, -x)
            np.random.uniform(-0.05, 0.05),  # left y
            np.random.uniform(-0.01, 0.06),  # right x (mostly outward, +x)
            np.random.uniform(-0.05, 0.05),  # right y
        ]
    )
    target = home + offset
    for _ in range(n):
        obs = env._env.task.get_observation(env._env.physics)
        curr_puppet_joint = obs["qpos"][:14]
        joint_actions, _ = trajectory_to_joint_actions(
            target,
            world_t_bases,
            kin_helper,
            curr_puppet_joint,
            curr_vel,
            dt,
            k_p,
            k_v,
            acc_lim,
            vel_lim,
        )
        env.step(joint_actions)


def rotation_success(init_pose, final_pose, actions) -> tuple[bool, float]:
    """True if the T stayed flat and rotated ~90 deg clockwise. Returns
    (success, achieved_delta_angle)."""
    table_normal = np.array([0, 0, 1])
    init_flat = np.dot(init_pose[:3, 2], table_normal) > 0.95
    final_flat = np.dot(final_pose[:3, 2], table_normal) > 0.95
    delta = np.linalg.inv(init_pose) @ final_pose
    delta_angle = float(np.arctan2(delta[1, 0], delta[0, 0]))
    ok = (
        init_flat
        and final_flat
        and -ACCEPT_MAX <= delta_angle <= -ACCEPT_MIN  # 80..105 deg clockwise
        and actions_in_range(actions)
    )
    return ok, delta_angle


# ----------------------------------------------------------------------------- #
# Main collection loop
# ----------------------------------------------------------------------------- #
@click.command()
@click.option("--output_dir", "-o", default="datasets/rotate_t")
@click.option("--n_episodes", "-n", default=100, type=int)
@click.option("--headless", "-h", is_flag=True)
@click.option("--seed", default=0, type=int)
@click.option("--x_min", default=-0.08, type=float)
@click.option("--x_max", default=0.08, type=float)
@click.option("--y_min", default=-0.08, type=float)
@click.option("--y_max", default=0.08, type=float)
@click.option("--debug_dir", default=None, help="If set, save every trial (incl. failures) as an mp4 here.")
@click.option("--no_settle", is_flag=True,
              help="Skip the random arm-settle step so the arm starts at a fixed home pose "
                   "(use with a fixed T position for a maximally-consistent, easier task).")
@click.option("--invisible_t", is_flag=True,
              help="Render the T invisible (alpha=0) while keeping its physics: the demos "
                   "show only the arms, but they really are rotating the (unseen) block.")
def main(output_dir, n_episodes, headless, seed, x_min, x_max, y_min, y_max, debug_dir,
         no_settle, invisible_t):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

    # Force upright + random-XY spawns.
    gae.sample_pusht_pose = make_upright_pose_sampler((x_min, x_max), (y_min, y_max))

    kin_helper = KinHelper(robot_name="trossen_vx300s")
    env = AlohaEnv("pusht")
    cv2.setNumThreads(1)

    dt = 1 / 10.0
    k_p, k_v = 50, 10
    acc_lim, vel_lim = 10.0, 0.04
    config = TrajectoryConfig(noise_std=0.001)
    coordinator = BimanualCoordination(config)

    episode_id = len(
        [p for p in Path(output_dir).glob("episode_*.hdf5")]
    )
    init_episode_id = episode_id
    trial = 0
    target_total = episode_id + n_episodes

    print(f"Collecting {n_episodes} rotate-T demos into {output_dir} "
          f"(starting at episode_{episode_id})")

    while episode_id < target_total:
        ep_seed = seed * 1_000_000 + trial
        trial += 1
        env.reset(seed=ep_seed)
        if invisible_t:
            hide_t(env)  # rendering-only; re-applied per reset in case of reload
        stabilize_t(env)  # let the T fall/settle before observing anything

        # world_t_bases (constant for the episode)
        obs = env._env.task.get_observation(env._env.physics)
        lb = pose_convert(obs["left_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
        rb = pose_convert(obs["right_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
        world_t_bases = np.stack([lb, rb])

        curr_vel = np.zeros(6)
        # Settle: drive the arms to a random ready pose so the gripper start varies
        # across demos. This can nudge the T, so we re-check uprightness below and
        # discard the trial if the T was bumped off upright. Skipped with --no_settle,
        # which leaves the arm at its fixed reset home (for the fixed-init easy task).
        if not no_settle:
            settle_arms(env, world_t_bases, kin_helper, curr_vel, dt, k_p, k_v,
                        acc_lim, vel_lim)
        curr_vel[:] = 0.0  # clean velocity state for the rotation PID

        episode = init_episode()
        init_angle = t_angle(env)
        if abs(init_angle) > UPRIGHT_TOL:
            print(f"  [skip] trial {trial}: settle bumped T to "
                  f"{np.degrees(init_angle):+.1f} deg -- discarded (not upright)")
            continue

        # Closed-loop clockwise rotation until the T has actually turned 90 deg.
        n_sub = 0
        while n_sub < MAX_SUBROTATIONS:
            cur_angle = t_angle(env)
            remaining = cur_angle - TARGET_ANGLE  # >0 means more CW rotation needed
            if remaining <= ANGLE_TOL:
                break
            # Fail fast on bad contact: not enough CW progress, or motion the wrong way.
            progress = init_angle - cur_angle  # >0 means we have rotated CW so far
            if n_sub >= 2 and progress < MIN_PROGRESS:
                break
            if cur_angle > init_angle + 0.15:  # T is being pushed CCW -- give up
                break
            step_angle = float(min(remaining, MAX_SUBSTEP))

            obs = env._env.task.get_observation(env._env.physics)
            t_info = extract_t_info(obs["env_state"])
            t_analyzer = TGeometryAnalyzer(t_info)
            current_arm_xy = get_current_arm_positions(obs, kin_helper, world_t_bases)
            traj = plan_cw_subrotation(
                t_analyzer, current_arm_xy, step_angle, config, coordinator
            )
            episode = execute_trajectory(
                env, traj, episode, world_t_bases, kin_helper, curr_vel, dt,
                k_p, k_v, acc_lim, vel_lim, True, headless, episode_id,
            )
            n_sub += 1

        # Evaluate + save.
        if len(episode["action"]) == 0:
            print(f"  trial {trial}: no motion recorded, skipping")
            continue
        init_pose = env_state_to_mat(episode["env_state"][0])
        final_pose = env_state_to_mat(episode["env_state"][-1])
        actions = np.stack(episode["action"])
        ok, achieved = rotation_success(init_pose, final_pose, actions)
        deg = np.degrees(achieved)
        if debug_dir:
            frames = np.stack(episode["obs"]["images"]["top_pov"])
            tag = "ok" if ok else "fail"
            vw = cv2.VideoWriter(
                f"{debug_dir}/trial{trial:03d}_{tag}_{deg:+.0f}deg.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (frames.shape[2], frames.shape[1]),
            )
            for fr in frames:
                vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            vw.release()
        if ok:
            save_episode(episode, output_dir, episode_id)
            episode_id += 1
            print(f"  [OK]   episode_{episode_id - 1}: rotated {deg:+.1f} deg "
                  f"in {n_sub} sub-rotations, {len(actions)} frames "
                  f"({episode_id - init_episode_id}/{n_episodes})")
        else:
            print(f"  [fail] trial {trial}: rotated {deg:+.1f} deg "
                  f"(init_angle={np.degrees(init_angle):+.1f}), {n_sub} subs -- discarded")

    print(f"Done. {episode_id - init_episode_id} demos saved to {output_dir}, "
          f"success rate {(episode_id - init_episode_id) / max(trial, 1):.2f}")


if __name__ == "__main__":
    main()

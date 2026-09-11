"""AlohaChunkEnv, vendored VERBATIM from scripts/ppo_residual_rotate_t.py on the
(not yet merged to main) strong-expert-and-scaling-v3 branch, commit e9d6a94 ("Add
the strong-expert pipeline: filtered self-imitation + residual PPO").

Why vendored rather than imported: this rlpd-imagined-rotate-t branch (and its
worktree) was created off origin/main per the plan doc's git-workflow guidance ("off
main, not off the in-progress strong-expert-and-scaling-v3, to avoid coupling to
unrelated in-flight work"). ppo_residual_rotate_t.py -- and hence AlohaChunkEnv --
only exists on that other, unmerged branch. Rather than importing across branches
(fragile, and would couple this branch to another session's in-progress work) or
re-deriving the class (risking the same kind of subtle discrepancy already found
between the WM env's reward and this precedent), this file is a byte-for-byte copy
of the class as of that commit. If strong-expert-and-scaling-v3 merges to main later,
this file should be deleted and real_sim_chunk_server.py switched to import the real
one, to avoid two copies drifting apart.

This is the reward AlohaChunkEnv.step_chunk actually computes (the already-proven
real-sim precedent this whole real-sim validation effort is trying to match exactly):
  prog = angle_prev - angle_end
  reward = prog*5.0 - 0.01 + (5.0 if (done and eval_success(init_pose, final_pose)) else 0)
where done = (raw angle crossed TARGET_ANGLE+ANGLE_TOL) or (step budget exhausted),
and eval_success requires BOTH the angle landing in [-105,-80] deg AND flatness
(surface normal within ~18 deg of vertical, both at init and at final) -- not just
"crossed -80 deg", unlike the WM env's simplified angle-only approximation.
"""
import collections

import numpy as np

import eval_dp_rotate_t as E
import collect_rotate_t as C
from gym_aloha.env import AlohaEnv
from yixuan_utilities.kinematics_helper import KinHelper
from sim_aloha_dataset_collection_scripted import (
    env_state_to_mat, trajectory_to_joint_actions,
)
from interactive_world_sim.utils.pose_utils import PoseType, pose_convert


class AlohaChunkEnv:
    """One MuJoCo instance stepped at action-chunk granularity."""

    def __init__(self, seed, mask, edges, max_steps, n_obs):
        self.env = AlohaEnv("pusht")
        self.kin = KinHelper(robot_name="trossen_vx300s")
        self.seed0, self.trial = seed, 0
        self.mask, self.edges = mask, edges
        self.max_steps, self.n_obs = max_steps, n_obs

    def _in_region(self):
        s = self.env._env.task.get_observation(self.env._env.physics)["env_state"]
        x, y = s[0] * 100, s[1] * 100
        e = self.edges
        if not (e[0] <= x < e[-1] and e[0] <= y < e[-1]):
            return False
        return bool(self.mask[int(np.digitize(x, e) - 1), int(np.digitize(y, e) - 1)])

    def reset(self):
        for _ in range(200):
            np.random.seed(self.seed0 + self.trial)
            self.env.reset(seed=self.seed0 + self.trial); self.trial += 1
            C.stabilize_t(self.env)
            o = self.env._env.task.get_observation(self.env._env.physics)
            lb = pose_convert(o["left_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            rb = pose_convert(o["right_base"][None], PoseType.POS_QUAT, PoseType.MAT)[0]
            self.wtb = np.stack([lb, rb])
            C.settle_arms(self.env, self.wtb, self.kin, np.zeros(6),
                          E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM)
            if abs(C.t_angle(self.env)) <= C.UPRIGHT_TOL and self._in_region():
                break
        self.init_pose = env_state_to_mat(
            self.env._env.task.get_observation(self.env._env.physics)["env_state"])
        img, ap, _ = E.get_obs(self.env, self.kin, self.wtb)
        self.img_h = collections.deque([img] * self.n_obs, maxlen=self.n_obs)
        self.ap_h = collections.deque([ap] * self.n_obs, maxlen=self.n_obs)
        self.steps, self.prev_ang = 0, C.t_angle(self.env)
        return np.stack(self.img_h), np.stack(self.ap_h)

    def step_chunk(self, chunk):
        """Execute one action chunk. Returns (obs, reward, done, info)."""
        curr_vel = np.zeros(6)
        reached = False
        for target_xy in chunk:
            o = self.env._env.task.get_observation(self.env._env.physics)
            joint, _ = trajectory_to_joint_actions(
                target_xy.astype(np.float64), self.wtb, self.kin, o["qpos"][:14],
                curr_vel, E.DT, E.K_P, E.K_V, E.ACC_LIM, E.VEL_LIM)
            self.env.step(joint)
            img, ap, _ = E.get_obs(self.env, self.kin, self.wtb)
            self.img_h.append(img); self.ap_h.append(ap)
            self.steps += 1
            if C.t_angle(self.env) <= C.TARGET_ANGLE + C.ANGLE_TOL:
                reached = True
            if self.steps >= self.max_steps:
                break
        ang = C.t_angle(self.env)
        # dense progress: CW rotation is negative, so a decrease in angle is progress
        prog = float(self.prev_ang - ang)
        self.prev_ang = ang
        done = reached or self.steps >= self.max_steps
        ok = False
        if done:
            final = env_state_to_mat(
                self.env._env.task.get_observation(self.env._env.physics)["env_state"])
            ok, _ = E.eval_success(self.init_pose, final)
        rew = prog * 5.0 - 0.01 + (5.0 if (done and ok) else 0.0)
        return (np.stack(self.img_h), np.stack(self.ap_h)), rew, done, {"success": ok}

"""PPO fine-tuning of the current best expert via a learned residual policy.

WHY A RESIDUAL. PPO needs log pi(a|s). A diffusion policy has no tractable action
likelihood -- the marginal over the denoising chain is intractable -- so PPO cannot
be applied to it directly. (The principled alternative, DPPO, reframes the denoising
chain itself as an MDP whose per-step transitions are Gaussian; that is a much larger
build.) Here the diffusion policy is FROZEN and used as a proposal: a small Gaussian
head learns a residual `delta` added to the proposed action chunk. Log-probs are then
exactly tractable, and delta = 0 reproduces the current expert, so the run is
warm-started at the expert's own performance by construction.

WHAT IT TARGETS. The measured failure mode is stalling, not overshooting: 9/10
baseline failures under-rotate and 8/10 exhaust the step budget, while successes
finish in ~448 of 800 steps. A residual nudge is well matched to breaking those
stalls.

ACTION SPACE. One 4-D residual per chunk decision (not per env step), clipped to
+-`--delta_max` metres and added to all n_action_steps actions of that chunk. So one
PPO step = one chunk = 8 env steps, giving ~50-100 PPO steps per episode instead of
~400-800.

REWARD. Dense progress toward the target rotation (the change in CW angle each
chunk), plus a success bonus, minus a small per-chunk time cost to discourage
stalling.

THROUGHPUT. The frozen expert keeps its full 100-step DDPM sampling so its behaviour
matches evaluation exactly; the speed comes from running `--n_envs` MuJoCo instances
in one process and batching their observations through the UNet in a single forward
pass, rather than from cutting denoising steps.

Usage:
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 python scripts/ppo_residual_rotate_t.py \
      --ckpt <best.ckpt> --n_envs 16 --iters 200 --out outputs/ppo_v1
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "data_collection"))
import eval_dp_rotate_t as E  # noqa: E402
import collect_rotate_t as C  # noqa: E402
import gym_aloha.env as gae  # noqa: E402
from gym_aloha.env import AlohaEnv  # noqa: E402
from yixuan_utilities.kinematics_helper import KinHelper  # noqa: E402
from sim_aloha_dataset_collection_scripted import (  # noqa: E402
    env_state_to_mat, trajectory_to_joint_actions,
)
from interactive_world_sim.utils.pose_utils import PoseType, pose_convert  # noqa: E402
from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402


# ----------------------------------------------------------------------------- env
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


# -------------------------------------------------------------------- frozen expert
class FrozenExpert:
    """Wraps the diffusion policy: obs batch -> (encoder features, action chunk)."""

    def __init__(self, ckpt, device):
        self.policy, self.n_obs, self.n_act = E.load_policy(ckpt, device)
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.device = device
        self.feat_dim = self.policy.obs_feature_dim * self.n_obs

    @torch.no_grad()
    def __call__(self, imgs, aps):
        obs = {"image": torch.as_tensor(imgs, device=self.device),
               "agent_pos": torch.as_tensor(aps, device=self.device)}
        nobs = self.policy.normalizer.normalize(obs)
        To = self.policy.n_obs_steps
        this = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        feat = self.policy.obs_encoder(this).reshape(imgs.shape[0], -1)
        chunk = self.policy.predict_action(obs)["action"]  # (B, n_act, 4)
        return feat, chunk


# ------------------------------------------------------------------ residual policy
class ResidualAC(nn.Module):
    """Gaussian residual actor + value head on the frozen encoder's features."""

    def __init__(self, feat_dim, chunk_dim, act_dim=4, hidden=256, log_std0=-1.0):
        super().__init__()
        d = feat_dim + chunk_dim
        self.body = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.mu = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)
        # start at the expert: zero-init the mean head so delta ~ 0 initially
        nn.init.zeros_(self.mu.weight); nn.init.zeros_(self.mu.bias)
        # Exploration scale. delta = tanh(a) * delta_max, so the residual's std is
        # roughly exp(log_std0) * delta_max: -3.0 gives ~0.05mm (too small to learn
        # from), -1.0 gives ~3.7mm, a nudge on the scale that breaks a stall.
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std0))

    def forward(self, feat, chunk):
        h = self.body(torch.cat([feat, chunk.flatten(1)], dim=-1))
        return self.mu(h), self.log_std.expand_as(self.mu(h)), self.v(h).squeeze(-1)

    def act(self, feat, chunk):
        mu, log_std, v = self(feat, chunk)
        d = torch.distributions.Normal(mu, log_std.exp())
        a = d.sample()
        return a, d.log_prob(a).sum(-1), v

    def evaluate(self, feat, chunk, a):
        mu, log_std, v = self(feat, chunk)
        d = torch.distributions.Normal(mu, log_std.exp())
        return d.log_prob(a).sum(-1), d.entropy().sum(-1), v


# --------------------------------------------------------------------------- ppo
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_envs", type=int, default=16)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--chunks_per_iter", type=int, default=32,
                    help="chunk decisions collected per env per PPO iteration")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent_coef", type=float, default=0.0)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--delta_max", type=float, default=0.01, help="metres")
    ap.add_argument("--log_std0", type=float, default=-1.0,
                    help="initial residual log-std (see ResidualAC)")
    ap.add_argument("--early_stop_drop", type=float, default=0.05,
                    help="stop once rolling success sits this far below the best seen")
    ap.add_argument("--early_stop_patience", type=int, default=12,
                    help="consecutive iterations below (best - drop) before stopping")
    ap.add_argument("--early_stop_after", type=int, default=30,
                    help="do not arm early stopping before this iteration. The rolling "
                         "window reads 100%% off a handful of episodes at the start and "
                         "then falls as it fills -- arming early would fire on that "
                         "warmup artefact, not on real degradation.")
    ap.add_argument("--snapshot_every", type=int, default=10,
                    help="also save residual_iter<N>.pt every N iterations. The v2 run "
                         "peaked around iter 40-45 and had DEGRADED by iter 80 (it "
                         "acquired wrong-direction failures), so the final model must "
                         "be chosen by evaluating snapshots -- not by taking the last "
                         "one, and not by the on-policy trace.")
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=70000)
    ap.add_argument("--feasible_mask",
                    default=str(Path(__file__).parent.parent / "datasets/feasible_mask_v3.json"))
    ap.add_argument("--x_min", type=float, default=-0.06)
    ap.add_argument("--x_max", type=float, default=0.06)
    ap.add_argument("--y_min", type=float, default=-0.06)
    ap.add_argument("--y_max", type=float, default=0.06)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = json.loads(Path(args.feasible_mask).read_text())
    mask, edges = np.array(m["mask"], bool), np.array(m["grid_cm"]["edges"])
    gae.sample_pusht_pose = C.make_upright_pose_sampler(
        (args.x_min, args.x_max), (args.y_min, args.y_max))

    expert = FrozenExpert(args.ckpt, device)
    envs = [AlohaChunkEnv(args.seed + 1000 * i, mask, edges, args.max_steps,
                          expert.n_obs) for i in range(args.n_envs)]
    print(f"{args.n_envs} envs, feat_dim={expert.feat_dim}, "
          f"chunk={expert.n_act}x4", flush=True)

    ac = ResidualAC(expert.feat_dim, expert.n_act * 4,
                    log_std0=args.log_std0).to(device)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)

    best_sr, best_iter, below = -1.0, -1, 0
    obs = [e.reset() for e in envs]
    imgs = np.stack([o[0] for o in obs]); aps = np.stack([o[1] for o in obs])
    ep_ret = np.zeros(args.n_envs); ep_succ, ep_done = [], []
    hist, t_start = [], time.time()

    for it in range(args.iters):
        B, T = args.n_envs, args.chunks_per_iter
        buf = {k: [] for k in ("feat", "chunk", "act", "logp", "val", "rew", "done")}
        for _ in range(T):
            feat, chunk = expert(imgs, aps)
            with torch.no_grad():
                a, logp, v = ac.act(feat, chunk)
            delta = torch.tanh(a) * args.delta_max          # bounded residual
            applied = (chunk + delta.unsqueeze(1)).cpu().numpy()
            buf["feat"].append(feat); buf["chunk"].append(chunk)
            buf["act"].append(a); buf["logp"].append(logp); buf["val"].append(v)
            rews, dones = np.zeros(B, np.float32), np.zeros(B, np.float32)
            for i, e in enumerate(envs):
                (im, apo), r, d, info = e.step_chunk(applied[i])
                rews[i], dones[i] = r, float(d)
                ep_ret[i] += r
                if d:
                    ep_succ.append(float(info["success"])); ep_done.append(ep_ret[i])
                    ep_ret[i] = 0.0
                    im, apo = e.reset()
                imgs[i], aps[i] = im, apo
            buf["rew"].append(torch.as_tensor(rews, device=device))
            buf["done"].append(torch.as_tensor(dones, device=device))

        with torch.no_grad():
            feat, chunk = expert(imgs, aps)
            _, _, last_v = ac(feat, chunk)
        # GAE
        val = torch.stack(buf["val"]); rew = torch.stack(buf["rew"])
        done = torch.stack(buf["done"])
        adv = torch.zeros_like(rew); gae_ = torch.zeros(B, device=device)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else val[t + 1]
            nonterm = 1.0 - done[t]
            delta_t = rew[t] + args.gamma * nv * nonterm - val[t]
            gae_ = delta_t + args.gamma * args.lam * nonterm * gae_
            adv[t] = gae_
        ret = adv + val
        f = torch.cat(buf["feat"]); ch = torch.cat(buf["chunk"])
        ac_t = torch.cat(buf["act"]); lp = torch.cat(buf["logp"])
        adv_f = adv.reshape(-1); ret_f = ret.reshape(-1)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        n = f.shape[0]
        for _ in range(args.epochs):
            for idx in torch.randperm(n, device=device).split(args.minibatch):
                nlp, ent, v = ac.evaluate(f[idx], ch[idx], ac_t[idx])
                ratio = (nlp - lp[idx]).exp()
                l1 = ratio * adv_f[idx]
                l2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_f[idx]
                loss = (-torch.min(l1, l2).mean()
                        + args.vf_coef * ((v - ret_f[idx]) ** 2).mean()
                        - args.ent_coef * ent.mean())
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()

        sr = float(np.mean(ep_succ[-100:])) if ep_succ else float("nan")
        rec = dict(iter=it, episodes=len(ep_succ), success_rate_last100=sr,
                   mean_return=float(np.mean(ep_done[-100:])) if ep_done else None,
                   log_std=float(ac.log_std.mean()), wall_s=int(time.time() - t_start))
        hist.append(rec)
        print(f"iter {it:3d}  eps={len(ep_succ):5d}  success(last100)={sr*100:5.1f}%  "
              f"ret={rec['mean_return']}  logstd={rec['log_std']:.2f}  "
              f"{rec['wall_s']}s", flush=True)
        (out / "history.json").write_text(json.dumps(hist, indent=1))
        torch.save({"ac": ac.state_dict(), "args": vars(args), "iter": it},
                   out / "residual_latest.pt")

        # Track the best rolling success and keep that snapshot. The v2 run peaked
        # near iter 45 and then DEGRADED (it lost the stall-breaking behaviour and
        # picked up wrong-direction failures), so the last iterate is not the model
        # to keep. Early stopping needs a full rolling window and a warmup guard,
        # otherwise it triggers on the small-sample 100% at the start.
        armed = (it >= args.early_stop_after and len(ep_succ) >= 100)
        if armed and sr == sr:
            if sr > best_sr:
                best_sr, best_iter, below = sr, it, 0
                torch.save({"ac": ac.state_dict(), "args": vars(args), "iter": it},
                           out / "residual_best.pt")
            elif sr < best_sr - args.early_stop_drop:
                below += 1
                if below >= args.early_stop_patience:
                    print(f"EARLY STOP at iter {it}: rolling success {sr*100:.1f}% has "
                          f"been >{args.early_stop_drop*100:.0f} pts below the best "
                          f"({best_sr*100:.1f}% at iter {best_iter}) for "
                          f"{below} iterations", flush=True)
                    break
            else:
                below = 0
        if args.snapshot_every and it % args.snapshot_every == 0:
            torch.save({"ac": ac.state_dict(), "args": vars(args), "iter": it},
                       out / f"residual_iter{it}.pt")


if __name__ == "__main__":
    main()

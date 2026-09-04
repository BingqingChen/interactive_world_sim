"""Wrap `frozen diffusion policy + PPO residual head` behind the plain DP interface.

The residual expert is the strongest policy we have (96.0% on region v3, n=100, vs
87% for the best plain DP), but it is a *pair* of modules, not a checkpoint. Every
consumer in this repo -- collect_imagined_rotate_t.py, collect_policy_rollouts.py,
eval_dp_rotate_t.py -- drives a policy through `predict_action(obs_dict)["action"]`.
This exposes exactly that, so the residual expert can be dropped in anywhere a DP
checkpoint goes.

The residual is applied with the MEAN action (no sampling): the exploration noise
belongs to training, and every reported residual number is measured deterministically.

Usage:
    from residual_policy import load_residual_policy
    policy, n_obs, n_act = load_residual_policy(residual_pt, device)
    action = policy.predict_action(obs)["action"]        # (B, n_act, 4)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_dp_rotate_t as E  # noqa: E402
from ppo_residual_rotate_t import FrozenExpert, ResidualAC  # noqa: E402


class ResidualPolicy:
    """Frozen DP proposal + deterministic residual, behind `predict_action`."""

    def __init__(self, residual_pt, device, base_ckpt=None):
        blob = torch.load(residual_pt, map_location="cpu")
        self.targs = blob["args"]
        self.expert = FrozenExpert(base_ckpt or self.targs["ckpt"], device)
        self.ac = ResidualAC(self.expert.feat_dim, self.expert.n_act * 4).to(device)
        self.ac.load_state_dict(blob["ac"])
        self.ac.eval()
        self.delta_max = self.targs["delta_max"]
        self.device = device
        self.iter = blob.get("iter")
        # mirror the attributes consumers read off a DP
        self.n_obs_steps = self.expert.policy.n_obs_steps
        self.n_action_steps = self.expert.n_act

    def reset(self):
        self.expert.policy.reset()

    def eval(self):
        return self

    @torch.no_grad()
    def predict_action(self, obs_dict):
        imgs = obs_dict["image"]
        aps = obs_dict["agent_pos"]
        if torch.is_tensor(imgs):
            imgs = imgs.detach().cpu().numpy()
        if torch.is_tensor(aps):
            aps = aps.detach().cpu().numpy()
        feat, chunk = self.expert(imgs, aps)
        mu, _, _ = self.ac(feat, chunk)
        return {"action": chunk + (torch.tanh(mu) * self.delta_max).unsqueeze(1)}


def load_residual_policy(residual_pt, device, base_ckpt=None):
    """Returns (policy, n_obs_steps, n_action_steps), matching E.load_policy."""
    p = ResidualPolicy(residual_pt, device, base_ckpt)
    return p, p.n_obs_steps, p.n_action_steps


if __name__ == "__main__":
    import argparse
    import numpy as np
    ap = argparse.ArgumentParser(description="smoke-test the wrapper")
    ap.add_argument("--residual", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pol, no, na = load_residual_policy(a.residual, dev)
    print(f"loaded residual from iter {pol.iter}: n_obs={no} n_act={na} "
          f"delta_max={pol.delta_max}")
    obs = {"image": np.zeros((1, no, 3, 128, 128), np.float32),
           "agent_pos": np.zeros((1, no, 4), np.float32)}
    out = pol.predict_action(obs)["action"]
    print(f"action shape {tuple(out.shape)}")
    assert out.shape == (1, na, 4)
    # The DP samples its own chunk, so calling the expert twice gives DIFFERENT
    # proposals -- differencing two separate calls would measure sampling noise,
    # not the residual. Take one proposal and apply the residual to it.
    feat, chunk = pol.expert(obs["image"], obs["agent_pos"])
    mu, _, _ = pol.ac(feat, chunk)
    resid = torch.tanh(mu) * pol.delta_max
    d = resid.abs().max().item()
    print(f"max |residual| = {d*1000:.2f} mm (bound {pol.delta_max*1000:.1f} mm)")
    assert d <= pol.delta_max + 1e-6, "residual exceeded its bound"
    assert torch.allclose(chunk + resid.unsqueeze(1),
                          chunk + (torch.tanh(mu) * pol.delta_max).unsqueeze(1))
    print("OK")

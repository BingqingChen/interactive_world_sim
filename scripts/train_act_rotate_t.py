"""Train ACT (Action Chunking Transformer) on the rotate-T demos, for a head-to-head
against Diffusion Policy on the identical task/obs/action setup.

Uses the real ACT model (tonyzhaozh/act, patched for state_dim=4 and torchvision>=0.15),
but with a custom variable-length dataset (ACT's own utils.py assumes fixed episode length,
which our 180-600-step demos violate).

Obs = top_pov 128x128 image + current bimanual EE-xy (qpos, 4-d); action = target EE-xy (4-d);
same as the Diffusion Policy setup. Saves policy_last.ckpt, policy_best.ckpt, stats.pkl.

Usage:
  python scripts/train_act_rotate_t.py --data_dir datasets/rotate_t \
    --ckpt_dir outputs/act_rotate_t --num_epochs 2000 --chunk_size 100
"""
import argparse
import glob
import os
import pickle
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ACT_ROOT = "/home/jacobhb/projects/worth_doing/act"
sys.path.insert(0, ACT_ROOT)
sys.path.insert(0, ACT_ROOT + "/detr")

CAM = "top_pov"


def qpos_from_ee(ee):  # ee: (...,2,4,4) -> (...,4) bimanual EE xy
    return ee[..., :2, 3].reshape(*ee.shape[:-3], 4)


def compute_stats(files):
    qs, acts = [], []
    for p in files:
        with h5py.File(p, "r") as f:
            qs.append(qpos_from_ee(f["obs/ee_pos"][:]).astype(np.float32))
            acts.append(f["action"][:].astype(np.float32))
    q = np.concatenate(qs); a = np.concatenate(acts)
    return {
        "qpos_mean": q.mean(0), "qpos_std": np.clip(q.std(0), 1e-2, None),
        "action_mean": a.mean(0), "action_std": np.clip(a.std(0), 1e-2, None),
    }


class RotateTACTDataset(Dataset):
    """One random timestep per __getitem__; returns a fixed chunk_size action window."""
    def __init__(self, files, chunk_size, stats, oversample=10):
        self.files = files
        self.chunk = chunk_size
        self.stats = stats
        self.oversample = oversample

    def __len__(self):
        return len(self.files) * self.oversample

    def __getitem__(self, idx):
        p = self.files[idx % len(self.files)]
        with h5py.File(p, "r") as f:
            T = f["action"].shape[0]
            t = np.random.randint(T)
            img = f[f"obs/images/{CAM}"][t]  # (128,128,3) uint8
            qpos = qpos_from_ee(f["obs/ee_pos"][t]).astype(np.float32)  # (4,)
            act = f["action"][t:t + self.chunk].astype(np.float32)  # (k,4)
        k = act.shape[0]
        padded = np.zeros((self.chunk, 4), np.float32)
        padded[:k] = act
        is_pad = np.ones(self.chunk, dtype=bool)
        is_pad[:k] = False
        qpos = (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        padded = (padded - self.stats["action_mean"]) / self.stats["action_std"]
        image = (np.moveaxis(img, -1, 0).astype(np.float32) / 255.0)[None]  # (1,3,128,128)
        return (torch.from_numpy(image), torch.from_numpy(qpos.astype(np.float32)),
                torch.from_numpy(padded), torch.from_numpy(is_pad))


def build_policy(chunk_size):
    saved = sys.argv
    sys.argv = ["x", "--ckpt_dir", "/tmp/x", "--policy_class", "ACT",
                "--task_name", "sim_rotate_t", "--seed", "0", "--num_epochs", "1"]
    from policy import ACTPolicy
    cfg = dict(lr=1e-5, num_queries=chunk_size, kl_weight=10, hidden_dim=512,
               dim_feedforward=3200, lr_backbone=1e-5, backbone="resnet18",
               enc_layers=4, dec_layers=7, nheads=8, camera_names=[CAM], state_dim=4)
    policy = ACTPolicy(cfg)
    sys.argv = saved
    return policy, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="datasets/rotate_t")
    ap.add_argument("--ckpt_dir", default="outputs/act_rotate_t")
    ap.add_argument("--num_epochs", type=int, default=2000)
    ap.add_argument("--chunk_size", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data_dir, "episode_*.hdf5")),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(files))
    n_val = max(1, int(len(files) * args.val_ratio))
    val_files = [files[i] for i in perm[:n_val]]
    train_files = [files[i] for i in perm[n_val:]]
    print(f"{len(train_files)} train / {len(val_files)} val episodes")

    stats = compute_stats(train_files)
    with open(os.path.join(args.ckpt_dir, "stats.pkl"), "wb") as f:
        pickle.dump(stats, f)

    train_ds = RotateTACTDataset(train_files, args.chunk_size, stats)
    val_ds = RotateTACTDataset(val_files, args.chunk_size, stats, oversample=20)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=6, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    policy, cfg = build_policy(args.chunk_size)
    with open(os.path.join(args.ckpt_dir, "policy_config.pkl"), "wb") as f:
        pickle.dump(cfg, f)
    policy.cuda()
    opt = policy.configure_optimizers()

    best_val = float("inf")
    for epoch in range(args.num_epochs):
        policy.train()
        tr = []
        for image, qpos, action, is_pad in train_ld:
            image, qpos, action, is_pad = (image.cuda(), qpos.cuda(), action.cuda(), is_pad.cuda())
            ld = policy(qpos, image, action, is_pad)
            ld["loss"].backward(); opt.step(); opt.zero_grad()
            tr.append(ld["loss"].item())
        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            policy.eval(); vl = []
            with torch.inference_mode():
                for image, qpos, action, is_pad in val_ld:
                    image, qpos, action, is_pad = (image.cuda(), qpos.cuda(), action.cuda(), is_pad.cuda())
                    vl.append(policy(qpos, image, action, is_pad)["loss"].item())
            v = float(np.mean(vl))
            print(f"epoch {epoch:4d}  train_loss {np.mean(tr):.4f}  val_loss {v:.4f}", flush=True)
            if v < best_val:
                best_val = v
                torch.save(policy.state_dict(), os.path.join(args.ckpt_dir, "policy_best.ckpt"))
        torch.save(policy.state_dict(), os.path.join(args.ckpt_dir, "policy_last.ckpt"))
    print(f"done. best_val={best_val:.4f}  ckpts in {args.ckpt_dir}")


if __name__ == "__main__":
    main()

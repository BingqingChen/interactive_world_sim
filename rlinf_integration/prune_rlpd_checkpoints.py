"""Keep only the top-K checkpoints of an RLinf run, ranked by in-training eval.

RLinf's embodied runner writes a checkpoint every `save_interval` steps, and each
one carries the full SAC replay buffer (~13 GB for the rotate-T RLPD configs) next
to ~90 MB of weights, optimizer, alpha and target-critic state. For these
experiments the result is the tensorboard metrics plus the best few policies, so
this deletes:
  * every checkpoint outside the top-K by `--metric`, and
  * the replay buffer inside the kept checkpoints (it only serves resuming a run).

`eval/<metric>` is logged at step index `global_step - 1` by the same
`_maybe_eval_and_checkpoint` call that writes `checkpoints/global_step_<global_step>`.
A checkpoint is only touched once it has an eval value and none of its files changed
for `--quiet_s` seconds, so a save still in progress is never pruned.

Usage:
  python prune_rlpd_checkpoints.py --run_dir ~/RLinf/results/<run> --dry_run
  python prune_rlpd_checkpoints.py --run_dir ~/RLinf/results/<run> --watch_pid <trainer pid>
"""

import argparse
import glob
import json
import os
import shutil
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_metric(run_dir, tag):
    """Map checkpoint global_step -> eval value."""
    vals = {}
    pattern = os.path.join(run_dir, "tensorboard", "events.out.tfevents.*")
    for f in sorted(glob.glob(pattern)):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        if tag in ea.Tags()["scalars"]:
            for e in ea.Scalars(tag):
                vals[e.step + 1] = float(e.value)
    return vals


def newest_mtime(path):
    newest = os.path.getmtime(path)
    for root, _, files in os.walk(path):
        for name in files:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, name)))
            except FileNotFoundError:
                pass
    return newest


def dir_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except FileNotFoundError:
                pass
    return total / 1e9


def prune_once(args, quiet_s):
    metric = load_metric(args.run_dir, args.metric)
    ckpts = {
        int(p.rsplit("_", 1)[1]): p
        for p in glob.glob(
            os.path.join(args.run_dir, "*", "checkpoints", "global_step_*")
        )
    }
    ranked = sorted((s for s in ckpts if s in metric), key=lambda s: (metric[s], s))
    keep = set(ranked[-args.keep :])
    now = time.time()
    events = []
    for step in sorted(ckpts):
        path = ckpts[step]
        if step not in metric:
            continue
        if now - newest_mtime(path) < quiet_s:
            continue
        if step in keep:
            targets = glob.glob(os.path.join(path, "*", "sac_components", "replay_buffer"))
            action = "keep_drop_replay_buffer"
        else:
            targets = [path]
            action = "delete_checkpoint"
        for t in targets:
            gb = dir_gb(t)
            events.append(
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "global_step": step,
                    args.metric: metric[step],
                    "action": action,
                    "path": t,
                    "gb": round(gb, 2),
                    "dry_run": args.dry_run,
                }
            )
            print(f"{action:24s} step {step:5d} {args.metric}={metric[step]:.3f} {gb:7.2f} GB  {t}")
            if not args.dry_run:
                shutil.rmtree(t)
    if events and not args.dry_run:
        with open(os.path.join(args.run_dir, "checkpoint_prune_log.jsonl"), "a") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
    print(f"top-{args.keep} by {args.metric}: " + ", ".join(
        f"{s}={metric[s]:.3f}" for s in sorted(keep, key=lambda s: -metric[s])))
    return events


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--metric", default="eval/success")
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--quiet_s", type=float, default=600.0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--watch_pid", type=int, default=None)
    parser.add_argument("--interval_s", type=float, default=300.0)
    args = parser.parse_args()

    if args.watch_pid is None:
        prune_once(args, args.quiet_s)
        return
    while pid_alive(args.watch_pid):
        prune_once(args, args.quiet_s)
        time.sleep(args.interval_s)
    # Trainer exited: its last save is complete, so no quiet period is needed.
    prune_once(args, 0.0)


if __name__ == "__main__":
    main()

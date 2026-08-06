"""Extract the published scaling-figure numbers from the per-run eval logs.

`outputs/random_init_evals/scale_eval_logs/` holds the full stdout of every
`eval_dp_rotate_t.py` run behind the scaling study (45 runs = 15 conditions x 3 seeds,
n=50 episodes each, seed 7000, final-ckpt rule). Each log's RESULTS block carries both
metrics plotted in `outputs/scaling_plot_final.png` (success rate) and
`outputs/scaling_plot_rotation.png` (mean final T rotation).

This writes them to a single JSON so the browser page can plot the real published
values rather than a re-run approximation.

Usage:
    python scripts/extract_scaling_metrics.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "outputs" / "random_init_evals" / "scale_eval_logs"
OUT_PATH = REPO_ROOT / "outputs" / "random_init_evals" / "scaling_metrics.json"

NAME_RE = re.compile(r"scale_eval_sc_r(\d+)i(\d+)_s(\d+)\.log$")
EPISODES_RE = re.compile(r"^episodes:\s+(\d+)", re.M)
SUCCESS_RE = re.compile(r"^SUCCESS RATE:\s+([0-9.]+)%", re.M)
ROT_RE = re.compile(r"^rotation deg:\s+mean=([+-]?[0-9.]+)\s+std=([0-9.]+)", re.M)
CONTACT_RE = re.compile(r"^contacts:\s+mean events=([0-9.]+)", re.M)


def parse_log(path: Path) -> dict:
    """Pull the RESULTS block out of one eval log."""
    text = path.read_text(errors="ignore")
    success, rot, episodes = (
        SUCCESS_RE.search(text),
        ROT_RE.search(text),
        EPISODES_RE.search(text),
    )
    if not (success and rot and episodes):
        raise ValueError(f"{path.name}: incomplete RESULTS block")
    contact = CONTACT_RE.search(text)
    return {
        "n_episodes": int(episodes.group(1)),
        "success_rate": float(success.group(1)),
        # The eval reports clockwise rotation as negative; the figure plots CW degrees
        # as a positive magnitude, so flip the sign here to match the published axis.
        "rotation_deg": -float(rot.group(1)),
        "rotation_std": float(rot.group(2)),
        "contact_events": float(contact.group(1)) if contact else None,
    }


def aggregate(runs: dict) -> dict:
    """Mean +/- standard error across seeds, per condition and per metric."""
    out = {}
    for (real, dose), seeds in sorted(runs.items()):
        entry = {"real": real, "dose": dose, "seeds": sorted(seeds), "per_seed": seeds}
        for metric in ("success_rate", "rotation_deg"):
            v = np.array([s[metric] for s in seeds.values()])
            entry[metric] = {
                "mean": round(float(v.mean()), 2),
                "sem": (
                    round(float(v.std(ddof=1) / np.sqrt(len(v))), 2)
                    if len(v) > 1
                    else 0.0
                ),
                "n_seeds": len(v),
            }
        entry["n_episodes"] = int(np.median([s["n_episodes"] for s in seeds.values()]))
        out[f"{real}_{dose}"] = entry
    return out


def main() -> None:
    """Parse every eval log and write the aggregated metrics JSON."""
    runs: dict = defaultdict(dict)
    logs = sorted(LOG_DIR.glob("scale_eval_sc_r*.log"))
    if not logs:
        raise SystemExit(f"no eval logs under {LOG_DIR}")
    for path in logs:
        m = NAME_RE.search(path.name)
        if not m:
            continue
        real, dose, seed = int(m[1]), int(m[2]), int(m[3])
        runs[(real, dose)][str(seed)] = parse_log(path)

    conditions = aggregate(runs)
    payload = {
        "source": "outputs/random_init_evals/scale_eval_logs/ (45 runs)",
        "protocol": "n=50 episodes, seed 7000, random init, final-ckpt rule",
        "figures": {
            "success_rate": "outputs/scaling_plot_final.png",
            "rotation_deg": "outputs/scaling_plot_rotation.png",
        },
        "conditions": conditions,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    n_runs = sum(len(v) for v in runs.values())
    print(f"parsed {n_runs} runs across {len(conditions)} conditions -> {OUT_PATH}\n")
    print(f"{'real':>5} {'dose':>5} {'seeds':>6}  {'success %':>16}  {'rot CW':>17}")
    for key in sorted(
        conditions, key=lambda k: (int(k.split("_")[0]), int(k.split("_")[1]))
    ):
        c = conditions[key]
        s, r = c["success_rate"], c["rotation_deg"]
        print(
            f"{c['real']:>5} {c['dose']:>5} {s['n_seeds']:>6}  "
            f"{s['mean']:>9.1f} +/-{s['sem']:>4.1f}  "
            f"{r['mean']:>10.1f} +/-{r['sem']:>4.1f}"
        )


if __name__ == "__main__":
    main()

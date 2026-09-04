# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive World Simulator (IWS) — a latent world model for robot manipulation that predicts future image observations conditioned on robot actions. Supports both simulated (MuJoCo/ALOHA) and real-world (ALOHA bimanual, R1 Lite) robot data. Based on [Diffusion Forcing](https://github.com/buoyancy99/diffusion-forcing).

## Environment Setup

```bash
mamba env create -f conda_env.yaml
conda activate iws
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126/
pip install -e .
# For MuJoCo sim environments:
git submodule update --init --recursive
uv pip install -e external/gym-aloha/
```

Set W&B entity before training:
```yaml
# configurations/config.yaml
wandb:
  entity: YOUR_WANDB_ENTITY
```

## Commands

**Lint / format:**
```bash
pre-commit run --all-files   # runs ruff, black, mypy, check-yaml, etc.
ruff check . --fix           # lint only
black .                      # format only
```

**Training (3-stage pipeline):**
```bash
# Stage 1: Autoencoder
python main.py +name=<run_name> algorithm=latent_world_model \
  experiment=exp_latent_dyn dataset=real_aloha_dataset \
  dataset.dataset_dir=data/mini/pusht \
  dataset.horizon=1 dataset.val_horizon=1 \
  dataset.obs_keys=[camera_1_color] \
  dataset.action_mode=bimanual_push \
  algorithm.latent_dim=512 algorithm.action_dim=4 \
  algorithm.training_stage=1

# Stage 2: Dynamics (requires stage 1 checkpoint)
python main.py +name=<run_name> algorithm=latent_world_model \
  ... algorithm.training_stage=2 \
  algorithm.load_ae="path/to/stage1.ckpt"

# Stage 3: Decoder finetuning (requires stage 2 checkpoint)
python main.py +name=<run_name> algorithm=latent_world_model \
  ... algorithm.training_stage=3 \
  algorithm.load_ae="path/to/stage2.ckpt"
```

**Inference (keyboard, no robot):**
```bash
python scripts/inference/teleoperate_keyboard.py \
  +output_dir='data/wm_demo' +use_joystick=false +use_dataset=false \
  +act_horizon=1 +scene=real \
  "+ckpt_paths=['outputs/pusht_cam1/checkpoints/best.ckpt']" \
  dataset=real_aloha_dataset \
  dataset.dataset_dir=data/mini/pusht/val \
  "dataset.obs_keys=['camera_1_color']"
```

**Local demo server:**
```bash
bash deploy/start_demo.sh                # world model only, serves its own UI (port 8000)
bash deploy/start_sim_demo.sh [PORT]     # lockstep sim-vs-WM BACKEND, API only (port 8001)
```
`start_sim_demo.sh` serves no HTML — the UI is the `../iws-demo-frontend` repo, run there
with `./serve.py`. `deploy/server.py` / `start_demo.sh` is the older self-contained demo
and is unaffected.

**Data collection (real robot):**
```bash
python scripts/data_collection/collect_real_aloha.py \
  --output_dir data/<task> --robot_sides right left \
  --frequency 10 --ctrl_mode bimanual_push --total_steps 200
python -m interactive_world_sim.real_world.robot_sleep --left --right
```

**Cluster (LSF/bsub):**
```bash
bsub < jobs/train_stage1.bsub
bsub < jobs/train_stage2.bsub
```

## Porting to another machine

`../diffusion_policy/docs/working_on_another_machine.md`. In short: ~77 tracked files
across the two repos hard-code `/home/jacobhb`, several script constants point at
specific checkpoints, and a residual `.pt` stores the absolute path of the frozen base
it loads (so that one is not greppable). Easiest route is to check both repos out at
the same absolute path. None of the datasets, checkpoints or `outputs/` are in git; the
science itself is ~65 MB of eval JSONs and videos, everything else is regenerable or
large.

## Data filing (project convention)

Authoritative rules live in the companion repo: `../diffusion_policy/docs/data_filing_rules.md`
(and `docs/archived_artifacts.md` for what has already been moved). The training
artifacts are produced here, so the rules bind this repo's scripts.

**The result is the eval JSON; the checkpoint is provenance.** A rotate-T checkpoint is
4.3 GB; the JSON holding its number is 18 KB. The root filesystem has hit 98% twice and
once aborted a 48-cell sweep mid-run, so accumulation — not any single file — is the
failure mode.

- **NVMe (`/`)** — training zarrs, pools, the current expert, `outputs/*/evals/*.json`,
  rollout videos.
- **`/data/storage/wm_archive/`** — evaluated checkpoints, kept indefinitely.
- **`/data/storage/trash/`** — superseded; delete outright only what is provably invalid.

`/data/storage` is a **spinning disk shared with another user** (191 MB/s vs 4.6 GB/s).
**Never put an actively-read training zarr there** — training reads shuffled image
chunks, near worst-case for that device.

Any sweep driver must, without being asked (`scripts/run_scaling_v3_sweep.py` is the
reference implementation):
1. delete each per-cell zarr after that cell trains;
2. **archive** that cell's checkpoints once its eval JSON exists — never delete; if the
   archive is unreachable, leave them in place and report it;
3. abort loudly on low free space (`MIN_FREE_GB`);
4. keep `topk k` at 1–2;
5. record `--n_videos 10` with a **per-cell `--video_dir`** — the eval default is a
   single shared path, so otherwise every cell overwrites the last. Episode *k* is the
   same initial state in every cell (`np.random.seed(seed + ep)`), so the clips are
   comparable across a grid.

**Check for load-bearing references before moving or deleting a checkpoint.** Paths
hide inside saved artifacts: `outputs/ppo_v2/residual_*.pt` stores the absolute path of
the frozen base checkpoint it loads at construction, so deleting that base silently
destroys the 96% expert. Grep the repos, grep `outputs/*/*.json`, and load any adapter
`.pt` to read the path inside it.

Cross-device moves: `rsync -a --remove-source-files` (unlinks only after a verified
transfer), never `mv`; record a size/mtime manifest before moving.

## Architecture

### Configuration System (Hydra)

All config is composed from YAML files under `configurations/`:
- `config.yaml` — root, selects experiment/dataset/algorithm/cluster
- `configurations/experiment/` — training loop settings (batch size, steps, val frequency)
- `configurations/algorithm/` — model hyperparameters (latent_dim, action_dim, diffusion params)
- `configurations/dataset/` — dataset path, horizon, obs_keys, action_mode, shape_meta

Override any config at the command line with Hydra syntax (`key=value`, `+key=value` for new keys, `"key=[a,b]"` for lists).

### Experiment / Training Loop

`main.py` → `BaseLightningExperiment` → PyTorch Lightning `Trainer`

`experiments/exp_base.py:BaseLightningExperiment` builds the dataset, dataloaders, algorithm, and runs `trainer.fit()`. `experiments/exp_latent_dyn.py:LatentDynExperiment` wires the compatible algorithm (`LatentWorldModel`) and datasets (`RealAlohaDataset`, `SimAlohaDataset`). Adding a new experiment means subclassing `BaseLightningExperiment`, registering in `compatible_algorithms`/`compatible_datasets`, and adding a YAML under `configurations/experiment/`.

### Model (`LatentWorldModel`)

`algorithms/latent_dynamics/latent_world_model.py` — the Lightning module for all three stages.

| Component | Location | Role |
|-----------|----------|------|
| Encoder | `_build_model()` — `nn.Sequential` of Conv2d + SiLU | RGB obs → compact latent |
| Decoder | `algorithms/models/cm_decoder.py:CMDecoder` | Latent → RGB via diffusion UNet (CMControlledUnetModel) |
| Dynamics | `algorithms/latent_dynamics/models/cm_latent_dynamics.py:CMLatentDynamics` | Predicts next latent from current latent + action; causal temporal attention |

Stage loading: `algorithm.load_ae` in the YAML points to a `.ckpt` file; the `LatentWorldModel` constructor loads encoder+decoder weights from it before training the next stage.

Checkpoints are saved alongside their Hydra config at `outputs/<date>/<time>/checkpoints/` with a `.hydra/config.yaml` sibling directory — the inference scripts rely on this layout to reload the model config.

### Data Pipeline

Raw data (HDF5 episodes) → Zarr cache → `ReplayBuffer` → `SequenceSampler` → DataLoader

`datasets/latent_dynamics/real_aloha_dataset.py:_convert_real_to_dp_replay()` converts per-episode HDF5 files into a single Zarr archive (`cache.zarr.zip`) on the first load. Subsequent runs use the cache.

**HDF5 episode layout:**
- Sim: `action (T,4)`, `obs/joint_pos (T,14)`, `obs/ee_pos (T,2,4,4)`, `obs/images/top_pov (T,128,128,3)`
- Real ALOHA: `action (T,D)`, `obs/joint_pos (T,14)`, `obs/ee_pos (T,2,4,4)`, `obs/images/camera_{0,1}_color (T,H,W,3)`

**DataLoader batch:**
```python
{
    "obs":  {"camera_0_color": (T, 3, 128, 128)},   # float32 [0,1]
    "goal": {"camera_0_color": (3, 128, 128)},
    "action": (T, D),                                # float32 [-1,1] normalized
}
```

`dataset.action_mode` and `dataset.action_dim` must be consistent — `bimanual_push` = 4-dim XY for both arms.

### Active Work: R1 Lite Integration

`docs/data-processing.md` documents the in-progress effort to harmonize R1 Lite MCAP data (mobile bimanual humanoid, 6 joints/arm, 4 cameras, mixed-rate ROS 2 topics) to IWS HDF5 format. Key deliverables tracked there:
- `scripts/data_collection/convert_r1lite_mcap_to_hdf5.py` (converter)
- `interactive_world_sim/datasets/latent_dynamics/r1lite_dataset.py` (new dataset class)
- `configurations/dataset/r1lite_dataset.yaml` (new config)

Start with `bimanual_push` action mode (EE XY, action_dim=4) for maximum model compatibility. Avoid ALOHA-specific gripper normalization functions (`PUPPET_GRIPPER_JOINT_NORMALIZE_FN`) and `KinHelper("trossen_vx300s")` in R1 Lite code.

### Real-World Interface

`real_world/` contains hardware drivers: `multi_realsense.py` / `single_realsense.py` (Intel RealSense), `aloha_bimanual_master.py` / `aloha_bimanual_puppet.py` (Interbotix ALOHA arms), `real_aloha_env.py` (environment wrapper). Camera calibration extrinsics are stored in `real_world/aloha_extrinsics/`.

### Deploy (Browser Demo)

`deploy/server.py` is a FastAPI + WebSocket server that runs inference from a pre-loaded `LatentWorldModel` and streams rendered frames to the browser. Start with `bash deploy/start_demo.sh`, then connect from the project page or `localhost`.

`deploy/sim_demo_server.py` is a second, self-contained demo that drives the **MuJoCo PushT sim and the world model in lockstep** on the same keyboard-commanded action stream, and renders ground truth / prediction / difference side by side with live PSNR and T-rotation drift (`deploy/sim_demo_metrics.py`; the UI lives in `../iws-demo-frontend`). Start with `bash deploy/start_sim_demo.sh` (API on port 8001 — over SSH, VS Code forwards the port automatically) and run the UI from the frontend repo. Notes:

- Both sides step **only while a key is held** (WASD = left arm, IJKL = right arm); release everything and the scene freezes. `R` resets, `Space` force-freezes.
- The world model is pure open-loop from the reset frame; **Re-sync WM** re-encodes the current sim frame into its latent history.
- Init modes: `fixed` (default), `random`, and `wm_train`. Only `wm_train` matches how the checkpoint's training data was collected (stock random-theta pose sampler + 100-step random arm init); the other two spawn the T upright and are measurably less faithful.
- On reset the sim is stepped `IWS_DEMO_SETTLE_STEPS` (100) times so the T falls from its 0.07 spawn height and comes to rest before anything is observed. In `fixed` mode the grippers are then driven outward by `IWS_DEMO_ARM_SPREAD` (0.06 m). That is not cosmetic: at the env's home pose the block rests at z=+0.017, propped on the grippers rather than on the table; once they move clear it settles to z=-0.001. The reachable spread is bounded by the workspace clip in `trajectory_to_joint_actions`, so values much beyond 0.06 have no further effect.
- Configured through `IWS_DEMO_*` env vars (uvicorn imports the app, so there is no argv). Defaults: `IWS_DEMO_ACTION_LAG=1` and `IWS_DEMO_DEC_STEPS=2`, both chosen by measurement — `dec_infer_steps=2` lifts the decoder's own reconstruction from ~27 dB to ~45 dB PSNR for +3 ms/tick, so the reported PSNR reflects dynamics error rather than decoder blur.
- `python deploy/sim_demo_server.py --ticks 100` exercises the whole engine headlessly (no browser), printing per-tick PSNR, both T angles and the timing breakdown. Steady state is ~45 ms/tick against the 100 ms budget.

**The browser UI lives in a separate repo** — `../iws-demo-frontend`. This repo is the backend: world model inference, the simulator, reset distributions. It serves `/ws`, `/api/scaling`, `/rollouts/...` and `/figures/{name}` with permissive CORS, and no HTML. Run the UI with `./serve.py` in the frontend repo; for the live demo, tunnel the backend with `ssh -N -L 8001:localhost:8001 <user>@<host>`. The split exists so rollout videos load from local disk on the user's own machine instead of trickling through the tunnel (measured at ~34 KB/s, which made a 10-video grid take ~50 s).

The frontend's scaling page plots the real-vs-imagined scaling results interactively and plays the rollouts behind each point:

- `scripts/extract_scaling_metrics.py` parses `outputs/random_init_evals/scale_eval_logs/` (45 runs = 15 conditions x 3 seeds, the full stdout of every `eval_dp_rotate_t.py` run behind the study) into `scaling_metrics.json`. It recovers **both** plotted metrics — success rate (`outputs/scaling_plot_final.png`) and mean T rotation (`outputs/scaling_plot_rotation.png`) — so the page shows the published n=50 numbers, not a re-run approximation. The eval reports CW rotation as negative; the extractor flips the sign to match the figure's axis.
- `scripts/collect_scaling_rollouts.py` records rollout videos per condition using the published protocol (seed 7000, random init), so episode *k* is the same initial state for every policy and matches episode *k* of the published eval. Videos are encoded at the true 10 Hz control rate, which is what lets the page's speed buttons be exact multiples of real time. The manifest is written incrementally, so the page is usable while the sweep runs.
- Keep the two straight: the **charts** are the published n=50 numbers; the **videos** are a separate small sample from one training seed. At `r=10, +0 imagined` (4% success) a 10-episode sample will usually show zero successes — that is expected, not a contradiction.
- `/api/scaling` merges both sources and is re-read per request; `/rollouts/...` serves the mp4s; `/figures/{success_rate,rotation_deg}` serves the original PNGs. The frontend prefers its own synced copies and falls back to these endpoints when it has not synced yet.
- `scripts/collect_wm_gallery.py` renders the world-model half of the page: 24 paired world-model-vs-simulator clips from `outputs/wm_quality/rollout_frames.npz` (same commanded actions, imagination left, physics right, per-episode PSNR/angle), and sampled episodes from the pooled imagined training set. Imagined samples are indexed by *pooled* index — randA(375)+randB(375)+randC(50) concatenated, exactly as `build_scaling_zarrs.py` slices it, where a `+N` dose takes pooled `0:N` — so each clip is tagged with the doses containing it. Pairs with `outputs/wm_video_metrics_plot.png`, whose per-episode data is in `outputs/wm_quality/video_metrics.npz`.
- `scripts/shrink_rollout_videos.py` re-encodes rollouts at the frames' native 128x128 and 5 fps (every 2nd control step, real-time duration preserved). Idempotent, so it is safe to re-run after a fresh collection sweep. Videos are ~157 KB each; without it they are ~890 KB.

# Invisible-cube ablation on robosuite Lift

## Question

Out-of-region demonstrations improve a policy's success on object positions its in-region demos
never covered. **Is that gain about spatial/motion coverage, or about seeing the object?**

The push-T study found that demos in which the manipulated object was *rendered invisible but
physically present* still helped (`no_t400_rand20` reached 34% against an 8.7% baseline at 20 real
demos). This repeats that ablation on a genuine 3D manipulation task — robosuite **Lift** — where
the answer can be read off a held-out region of cube placements.

## Answer

**On Lift, the gain requires seeing the object.** Hiding the cube removes essentially all of it.

| policy | training data | small region (±3 cm) | large region (±10 cm) |
|---|---|---|---|
| `centre` | 50 demos, cube in ±3 cm | 62.0% ± 6.9 | 46.0% ± 7.0 |
| `centre+visible` | + 400 demos, cube in ±10 cm | 70.0% ± 6.5 | **72.0% ± 6.3** |
| `centre+invisible` | + the same 400, cube hidden | 70.0% ± 6.5 | **48.0% ± 7.1** |

n = 50 per cell, all policies evaluated on the *same* fixed initial states, so contrasts are paired:

| contrast | region | difference |
|---|---|---|
| `+visible` − `centre` | large | **+26.0% ± 8.9 (+2.9 s.e.)** |
| `+invisible` − `centre` | large | +2.0% ± 7.8 (+0.3 s.e.) |
| `+invisible` − `+visible` | large | **−24.0% ± 8.4 (−2.9 s.e.)** |
| `+invisible` − `+visible` | small | +0.0% ± 8.1 (0.0 s.e.) |
| `+visible` − `centre` | small | +8.0% ± 9.0 (+0.9 s.e.) |
| `+invisible` − `centre` | small | +8.0% ± 7.5 (+1.1 s.e.) |

Out of region, the invisible set buys **+2 points** where the identical set with the cube visible
buys **+26**. `centre+invisible` is statistically indistinguishable from the 50-demo baseline.
**This does not replicate the push-T result.**

The `+invisible` − `+visible` row is the cleanest statement available: those two training sets have
**byte-identical** actions, states, episode boundaries and physics, and identical gradient-step
budgets. The only difference is whether the cube's pixels were drawn. So the −24 points out of
region is attributable to seeing the object and to nothing else.

### Secondary observation

In region, `+visible` and `+invisible` are *exactly* equal (70.0% each, paired difference
0.0% ± 8.1), and both sit ~8 points above `centre` — suggestively but not significantly.

A reading consistent with both rows: within ±3 cm the cube is nearly where the policy already
expects it, so extra data helps mainly as motion/robustness coverage, which the invisible demos
supply just as well. Out at ±10 cm the policy must actually *localize* the cube, and demos that
never showed it there carry no information for that. Motion coverage is not the bottleneck;
object localization is.

## Method

**Demos come from π₀.₅** (`pi05_libero`), served over a websocket and rolled out by
`openpi/examples/libero/main_lift.py`. Not scripted.

- Widening the placement region is one line: the stock `UniformRandomSampler` re-reads
  `x_range`/`y_range` on every `sample()`, so mutating them after `suite.make` is enough.
  Added as `--args.cube-range` (default 0.03 = stock).
- **Yield at ±10 cm: 403 successes from 1,436 rollouts = 28.1%**, against 32.7% at ±3 cm. It
  degrades gracefully with distance from centre (`|xy|_inf` in [0,0.03): 50%, [0.03,0.06): 33%,
  [0.06,0.11): 25%) rather than collapsing, so the ±10 cm region needed no shrinking.
- Initial states are exactly reproducible: robosuite has no `.seed()`, and the sampler draws from
  the *global* numpy RNG, so `np.random.seed(seed*100003 + idx)` before `env.reset()` pins an
  episode. That is also what makes collection workers independent — an episode's initial state is
  a pure function of its index, so two workers on disjoint index blocks need no coordination.

**The invisible set is a pure re-render, not a re-simulation.** Every step's full `sim_state`
(32-D: time + qpos + qvel) is stored, so `sim.set_state_from_flattened` + `forward()` reproduces
each instant exactly; hiding the cube is `geom_rgba[geom_name2id("cube_g0_vis")][3] = 0.0`, which
touches only the *visual* geom, leaving collision, physics and `_check_success()` untouched.
Verified per episode: all 17 non-image fields byte-identical, red cube pixels 99.8% (agentview) /
100% (wrist) removed, differences spatially localized.

**Policies** are Diffusion Policy (`DiffusionUnetHybridImagePolicy`), two cameras (agentview +
wrist at 128²), 8-D proprio (`eef_pos` + eef axis-angle + `gripper_qpos`), 7-D OSC_POSE delta
action, `horizon 16 / n_obs 2 / n_action 8`, `crop_shape [115,115]`. The wrist camera is not
optional: Lift cannot be grasped from agentview alone, and a policy that cannot grasp would leave
all three conditions near 0% and the comparison empty.

**Equal optimization budget** was enforced on measured `global_step`, not epoch count — the
450-demo sets have ~10× more samples per epoch, so a fixed epoch count would have handed them ~10×
the optimization. Evaluated checkpoints: `centre` 17,067 steps, `vis` and `invis` 17,749 each.

**Evaluation** restores 50 fixed post-settle states per region (sampled once, episode indices
≥ 100000, disjoint from every training seed) and rolls the policy out to 200 steps, scoring with
`env._check_success()`. The cube is always visible at evaluation, whatever the policy trained on.

## Limitations

- **A 4.0% budget asymmetry.** `vis`/`invis` got 17,749 gradient steps against `centre`'s 17,067,
  from an off-by-one: the checkpoint saved during epoch index `E` reflects `E+1` completed epochs.
  It favours the treatment arms, but it cannot explain the result — `centre`'s val_loss was already
  flat from epoch ~50 (best 0.0549 at epoch 50, 0.0580 at epoch 100, never better through 293), so
  it is not step-limited. The two treatment arms are exactly equal to each other, which is the
  contrast the headline rests on.
- **n = 50 per cell** gives ~7% per-cell s.e.m. The +26 and −24 point effects clear that
  comfortably (~2.9 s.e.); the ~+8 point in-region effects do not (~1 s.e.) and should be treated
  as suggestive only.
- **A single training seed per condition.** The push-T scaling study used 3. Seed variance is
  therefore not separated from condition effects, though a 24-point paired gap is large relative to
  the seed spread seen in the push-T runs.
- **The baseline is high.** 46% out-of-region leaves ~16 points of headroom to 62%, yet `+visible`
  reached 72% — above the in-region baseline — so the extra data did more than close a gap. The
  push-T analogue started from 8.7%, a very different regime, which is one reason the two results
  may legitimately differ rather than one being wrong.

## Reproducing

```bash
# 1. collect at +/-10 cm (needs a pi0.5 LIBERO server on $PORT; see run_lift_large.sh header)
cd ~/projects/openpi
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run scripts/serve_policy.py --env LIBERO --port 8020 &
START=60 COUNT=800 PORT=8020 EGL=1 bash run_lift_large.sh
examples/libero/.venv/bin/python tools/count_lift_successes.py     # stop at 400

# 2. renderer fidelity gate, then the invisible set  (IWS repo, .venv-lift)
.venv-lift/bin/python scripts/lift_invis/replay_lift.py --mode check \
    -i ~/projects/openpi/data/robosuite/rollouts_lift_pi/lift --n-episodes 3   # want > 35 dB
.venv-lift/bin/python scripts/lift_invis/replay_lift.py --mode invisible --successes-only \
    -i <collection dir> -o <.../lift_large_invis>
.venv-lift/bin/python scripts/lift_invis/verify_invisible.py <visible.npz> <invisible.npz>

# 3. datasets  (order matters: `vis` pins the manifest, `invis` refuses to run without it)
.venv/bin/python scripts/lift_invis/build_lift_zarrs.py --which centre
.venv/bin/python scripts/lift_invis/build_lift_zarrs.py --which vis
.venv/bin/python scripts/lift_invis/build_lift_zarrs.py --which invis
.venv/bin/python scripts/lift_invis/compute_epochs.py datasets/lift_vis_dp.zarr

# 4. train (diffusion_policy repo; needs no robosuite, so it can run on another box)
python train.py --config-name=train_diffusion_unet_hybrid_lift_{centre,vis,invis}

# 5. evaluate and report
.venv-lift/bin/python scripts/lift_invis/eval_lift_dp.py --ckpt <latest.ckpt> --label vis \
    --regions small large --n-episodes 50 --max-steps 200 \
    --out outputs/lift_invis/results_vis.json --video-dir outputs/lift_invis/videos
.venv/bin/python scripts/lift_invis/report_results.py outputs/lift_invis/results_*.json
```

### Files

| path | role |
|---|---|
| `openpi/examples/libero/main_lift.py` | collection; `--args.cube-range` widens the placement square |
| `openpi/run_lift_large.sh` | one collection worker, chunked at 250, resumable |
| `openpi/tools/count_lift_successes.py` | pooled success count, skips in-flight files |
| `scripts/lift_invis/replay_lift.py` | fidelity gate + invisible-set generator |
| `scripts/lift_invis/verify_invisible.py` | the three invisible-set assertions |
| `scripts/lift_invis/build_lift_zarrs.py` | npz → three DP zarrs, manifest-pinned |
| `scripts/lift_invis/compute_epochs.py` | equal-budget epoch / checkpoint_every |
| `scripts/lift_invis/sample_eval_inits.py` | the fixed 100 initial states (run once) |
| `scripts/lift_invis/eval_lift_dp.py` | closed-loop eval, in-process |
| `scripts/lift_invis/report_results.py` | final table + paired contrasts |
| `datasets/lift_invis_manifest.json` | the exact 50 + 400 episode indices used |
| `datasets/lift_invis_eval_inits.npz` | the 100 fixed initial states |
| `outputs/lift_invis/` | `results_*.json`, `final_table.txt`, 30 rollout videos |

## Gotchas worth not rediscovering

- **`MjSimState` has no `flat` setter.** `state.flat = ...` then `sim.set_state(state)` silently
  restores the *unmodified* state, so every frame renders the reset pose — which measures as
  ~16 dB against the stored frames and looks exactly like a renderer version mismatch. Use
  `sim.set_state_from_flattened(flat)` then `sim.forward()`.
- **DP only checkpoints when `epoch % checkpoint_every == 0`**, so the final epoch is usually never
  saved and `latest.ckpt` is some earlier epoch. Set `checkpoint_every = num_epochs - 1`.
- **Training cannot share a GPU with the JAX policy server**: it preallocates ~75% of the card
  (24.1 GiB of 31.4), so DP OOMs instantly. Use another GPU or
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- **`np.savez_compressed` appends `.npz`** to any path not already ending in it, so a temp file
  named `foo.npz.partial` lands at `foo.npz.partial.npz` and a subsequent rename fails.
- `np.savez_compressed` is also not atomic: a partially written episode raises `BadZipFile`, so
  anything scanning the collection directories during collection must catch it.
- **`.venv-lift` needs four pins** or the DP workspace will not import: `robosuite==1.4.1`,
  `mujoco==3.2.3`, `numcodecs==0.11.0` (0.16 breaks zarr 2.12), `diffusers==0.29.2` (0.39 breaks
  DP). Installing `robomimic==0.3.0` silently pulls torch 2.13 over the pinned 2.7.1+cu128 —
  re-pin torch *and* torchvision 0.22.1 afterwards.
- robosuite's EGL context raises a harmless `EGLError: EGL_NOT_INITIALIZED` from `__del__` at
  interpreter exit, *after* all real output. Grep for your own markers rather than tailing.
- Launching a background job over ssh hangs the ssh call even with `nohup ... &` (ssh waits on the
  inherited streams). The job does start — check before relaunching, or you get two copies.

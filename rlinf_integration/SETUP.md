# RLinf venv setup for the rotate-T RLPD integration

Verified working recipe for getting IWS's simulation stack (MuJoCo/dm_control/gym_aloha/
yixuan_utilities) to coexist with RLinf's own pinned ML stack (torch==2.11.0,
transformers==4.57.6, etc.) in one venv. Two ecosystems' pinned requirements conflict in
several places; this is the order that got a clean `from rlinf.envs import get_env_cls;
get_env_cls("iws_rotate_t_wm")` import working.

## 0. Base setup

```bash
git clone https://github.com/RLinf/RLinf.git ~/RLinf   # vanilla upstream, not a fork
cd ~/RLinf
rm -rf .venv
uv sync --python 3.11 --extra embodied   # NOT the default (3.13) -- dm_control's
                                          # labmaze dep has no cp313 wheel and needs
                                          # bazel (not installed) to build from source.
                                          # 3.11 matches IWS's own robodiff/uv env and
                                          # has prebuilt wheels for everything.
```//
`--extra embodied` (not `franka`/`agentic`/etc. -- these are declared mutually
conflicting in `pyproject.toml`'s `[tool.uv] conflicts`) pulls in gymnasium/gym/
torchvision/timm/transformers, needed for the embodiment RL path generally.

## 1. IWS simulation stack

```bash
V=~/RLinf/.venv/bin
IWS=/home/jacobhb/projects/worth_doing/interactive_world_sim   # or the worktree, see below
"$V/python" -m pip install opencv-python "zarr==2.18.7" "numcodecs<0.13" \
    "dm_control==1.0.27" "mujoco==3.3.0" "transforms3d==0.4.2" \
    -e "$IWS/external/gym-aloha" \
    "git+https://github.com/wangyixuan12/yixuan_utilities@805ea8bbb58eb2bf727d05970b7e287c99b82b42"
```

`pinocchio` is a trap: PyPI's `pinocchio==0.4.3` is an UNRELATED nose-testing-framework
plugin that happens to squat the same top-level import name as the real Pinocchio
rigid-body-dynamics library `yixuan_utilities` actually needs. Do not `pip install
pinocchio`. Instead copy the real, working install directly from IWS's own venv (both
are Python 3.11, same ABI, so a raw file copy is safe):

```bash
cp -r "$IWS/.venv/lib/python3.11/site-packages/pinocchio" \
      "$IWS/.venv/lib/python3.11/site-packages/pinocchio-0.4.3.dist-info" \
      ~/RLinf/.venv/lib/python3.11/site-packages/
```

Then the rest of `yixuan_utilities.kinematics_helper`'s import chain, each a normal
PyPI package (no traps):

```bash
"$V/python" -m pip install "sapien==2.2.2" urdfpy h5py matplotlib lightning
```

`urdfpy` regresses `pyopengl`/`networkx` to old pins that break `dm_control`'s EGL
renderer and torch respectively -- re-pin immediately after:

```bash
"$V/python" -m pip install "pyopengl==3.1.10" "networkx==3.6.1"
```

`lightning` (needed by IWS's `LatentWorldModel`, via `torchmetrics`/`transformers`)
drags in `huggingface-hub>=1.0`, which breaks RLinf's own `transformers==4.57.6`
(`huggingface-hub<1.0,>=0.34.0` required) -- re-pin back down last:

```bash
"$V/python" -m pip install "huggingface-hub<1.0,>=0.34.0"
```

## 2. IWS package itself + code/data root split

```bash
"$V/python" -m pip install --no-deps -e "$IWS"   # editable install, matches IWS's own convention
```

**Install `-e` from the worktree, not the main checkout**, if working on a branch --
`world_model_iws_rotate_t_env.py`'s `IWS_ROOT` points at whichever path is used here,
and any IWS-side code changes for this work (e.g. `est_angle_with_conf` added to
`collect_imagined_rotate_t.py`) only exist on the branch.

Git worktrees don't materialize gitignored content (`datasets/`, `ckpts/`, `outputs/`),
so if `IWS_ROOT` is a worktree, symlink the shared data dirs in from the main checkout:

```bash
WT=.../rlpd-imagined-rotate-t
for d in datasets ckpts outputs; do ln -s "$IWS_MAIN/$d" "$WT/$d"; done
```

## 3. Verify

```bash
cd ~/RLinf && MUJOCO_GL=egl ./.venv/bin/python -c "
from rlinf.envs import get_env_cls
print(get_env_cls('iws_rotate_t_wm'))
"
```

Expect a wall of "interactive-world-sim 0.1.0 requires X==Y, but you have X==Z"
`pip`-resolver warnings on the way -- these are advisory only (pip doesn't enforce
transitive pins at runtime) and are the expected byproduct of reconciling two
independently-pinned ML stacks in one venv, not import failures. The real version gap
worth tracking: **IWS's own pins call for `torch==2.7.1+cu128`, this venv has
`torch==2.11.0`** -- neither the import chain nor (per this project's own precedent of
running newer torch than pinned for Blackwell GPU support) is expected to break from
this alone, but it hasn't been verified against actual WM inference correctness yet --
that's the job of the end-to-end smoke test (plan doc's Verification section), not
this import-level check.

Full working package list snapshotted at `pip freeze` time: see job scratch dir
(`rlinf_venv_freeze.txt`) for exact resolved versions if this needs to be reproduced
exactly.

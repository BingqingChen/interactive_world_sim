#!/usr/bin/env bash
# Start the lockstep PushT demo: MuJoCo sim vs. the latent world model, side by side.
#
# Usage: bash deploy/start_sim_demo.sh [PORT]
#
# Then open http://localhost:<PORT> in your browser. Over SSH, VS Code forwards the
# port automatically; nothing else is needed. Drive with WASD (left arm) and IJKL
# (right arm) -- both the simulator and the world model step only while a key is held.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PORT="${1:-8001}"

# MUJOCO_GL must be set before the server imports mujoco, or dm_control falls back to
# glfw and dies on the missing DISPLAY.
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Runtime config (uvicorn imports the app, so these are read from the environment).
export IWS_DEMO_SEED="${IWS_DEMO_SEED:-900000}"
export IWS_DEMO_INIT="${IWS_DEMO_INIT:-fixed}"
export IWS_DEMO_IMG="${IWS_DEMO_IMG:-png}"
export IWS_DEMO_ACTION_LAG="${IWS_DEMO_ACTION_LAG:-1}"
export IWS_DEMO_DEC_STEPS="${IWS_DEMO_DEC_STEPS:-2}"

echo "Loading the world model and simulator (~10 s), then serving on http://localhost:${PORT}"
echo "Controls: WASD = left arm, IJKL = right arm, R = reset, Space = freeze."
echo ""

cd "$REPO_ROOT"

# Prefer the repo's own venv so the script works from an unactivated shell.
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi

# Loopback only: this is reached through the VS Code port forward, not the LAN.
# No --reload: re-importing after mujoco has loaded breaks the EGL context.
exec "$PY" -m uvicorn deploy.sim_demo_server:app --host 127.0.0.1 --port "$PORT"

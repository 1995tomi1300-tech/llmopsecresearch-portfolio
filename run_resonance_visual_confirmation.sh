#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${1:-}"
OUTPUT_DIR="${2:-/mnt/d/resonance_phase6_outputs/visual_confirmation}"
TARGET_NODES="${TARGET_NODES:-512}"
FPS="${FPS:-18}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/d/.venv_sensor/bin/python}"

cmd=("$PYTHON_BIN" "/mnt/d/resonance_visual_confirmation.py" "--output-dir" "$OUTPUT_DIR" "--target-nodes" "$TARGET_NODES" "--fps" "$FPS")
if [[ -n "$SCENARIO" ]]; then
  cmd+=("--scenario" "$SCENARIO")
fi

"${cmd[@]}"

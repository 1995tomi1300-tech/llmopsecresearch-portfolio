#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <telemetry_output_jsonl> <command> [args...]" >&2
  exit 1
fi

TELEMETRY_OUTPUT="$1"
shift

CPU_SET="${SPIRAL_CPU_SET:-29-31}"
INTERVAL="${SPIRAL_TELEMETRY_INTERVAL:-1.0}"
THREADS="${SPIRAL_NUMERIC_THREADS:-3}"

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

python3 "$ROOT_DIR/resonance_field/hardware_frequency_monitor.py" \
  --output "$TELEMETRY_OUTPUT" \
  --interval "$INTERVAL" \
  --count 1

python3 "$ROOT_DIR/resonance_field/hardware_frequency_monitor.py" \
  --output "$TELEMETRY_OUTPUT" \
  --interval "$INTERVAL" &
MONITOR_PID=$!

cleanup() {
  if kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

taskset -c "$CPU_SET" "$@"

python3 "$ROOT_DIR/resonance_field/hardware_frequency_monitor.py" \
  --output "$TELEMETRY_OUTPUT" \
  --interval "$INTERVAL" \
  --count 1

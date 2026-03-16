#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

OUTPUT_DIR="${1:-/mnt/d/resonance_phase16_dim_narrow_outputs}"
shift || true

mkdir -p "$OUTPUT_DIR"
TELEMETRY_JSONL="$OUTPUT_DIR/phase16_dim_narrow_hardware_frequency.jsonl"

bash "$ROOT_DIR/run_spiral_with_telemetry.sh" \
  "$TELEMETRY_JSONL" \
  python3 "$ROOT_DIR/resonance_phase16_dim_narrow.py" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-/mnt/d/resonance_phase16_scaled_outputs}"
shift || true

mkdir -p "$OUTPUT_DIR"
TELEMETRY_JSONL="$OUTPUT_DIR/phase16_scaled_hardware_frequency.jsonl"

bash /mnt/d/run_spiral_with_telemetry.sh \
  "$TELEMETRY_JSONL" \
  python3 /mnt/d/resonance_phase16_scaled.py \
  --output-dir "$OUTPUT_DIR" \
  "$@"

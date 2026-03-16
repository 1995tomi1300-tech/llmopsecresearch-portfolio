#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <output_json> <input...> [--logical-total N]" >&2
  exit 1
fi

OUTPUT_JSON="$1"
shift

OUT_DIR="$(dirname "$OUTPUT_JSON")"
mkdir -p "$OUT_DIR"
TELEMETRY_JSONL="$OUT_DIR/population_baseline_hardware_frequency.jsonl"

bash /mnt/d/run_spiral_with_telemetry.sh \
  "$TELEMETRY_JSONL" \
  python3 /mnt/d/resonance_population_baseline.py \
  "$@" \
  --output "$OUTPUT_JSON"

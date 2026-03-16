#!/usr/bin/env bash
set -euo pipefail

PAYLOAD="${1:-/mnt/d/nvda_virtualis_ter_lidar/outputs/synthetic_lidar_bridge_payload.json}"
OUTPUT_DIR="${2:-/mnt/d/nvda_virtualis_ter_lidar/outputs/dual_gpu_batch}"

python3 /mnt/d/nvda_virtualis_ter_lidar/generate_synthetic_lidar_bridge.py --output "$PAYLOAD" >/dev/null
python3 /mnt/d/resonance_spiral_dual_gpu_batch.py "$PAYLOAD" --output-dir "$OUTPUT_DIR"

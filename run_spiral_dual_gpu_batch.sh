#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <report.json> [more_report.json ...]" >&2
  exit 1
fi

python3 /mnt/d/resonance_spiral_dual_gpu_batch.py "$@"

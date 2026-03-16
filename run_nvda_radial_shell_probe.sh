#!/usr/bin/env bash
set -euo pipefail

OUTPUT_TXT="${1:-/mnt/d/nvda_virtualis_ter_lidar/outputs/max_radial_shell_probe.txt}"
OUTPUT_JSON="${2:-/mnt/d/nvda_virtualis_ter_lidar/outputs/max_radial_shell_probe.json}"

mkdir -p "$(dirname "$OUTPUT_TXT")"
mkdir -p "$(dirname "$OUTPUT_JSON")"

docker run --rm --gpus all --entrypoint /bin/bash \
  -v /mnt/d:/workspace -w /workspace \
  modular/max-nvidia-full:latest \
  -lc "mojo /workspace/nvda_virtualis_ter_lidar/max_kernels_scaffold/bench_radial_shell_focus.mojo" \
  | tee "$OUTPUT_TXT"

python3 /mnt/d/nvda_virtualis_ter_lidar/parse_mojo_benchmark_report.py \
  --input "$OUTPUT_TXT" \
  --output "$OUTPUT_JSON"

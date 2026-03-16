#!/usr/bin/env bash
set -euo pipefail

OUTPUT_FILE="${1:-/mnt/d/nvda_virtualis_ter_lidar/outputs/max_kernel_probe.txt}"
mkdir -p "$(dirname "$OUTPUT_FILE")"

wsl.exe -d Ubuntu -u root -- bash -lc \
  'docker run --rm --gpus all --entrypoint /bin/bash -v /mnt/d:/workspace -w /workspace modular/max-nvidia-full:latest -lc "mojo /workspace/nvda_virtualis_ter_lidar/max_kernels_scaffold/bench_spatial_reduction.mojo"' \
  | tee "$OUTPUT_FILE"

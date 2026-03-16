#!/usr/bin/env bash
set -euo pipefail

CPU_SET="${SPIRAL_CPU_SET:-29-31}"
exec taskset -c "$CPU_SET" python3 /mnt/d/spiral_control_mcp/server.py

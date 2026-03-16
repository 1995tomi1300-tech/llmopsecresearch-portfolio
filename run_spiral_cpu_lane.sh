#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 1
fi

CPU_SET="${SPIRAL_CPU_SET:-29-31}"
exec taskset -c "$CPU_SET" "$@"

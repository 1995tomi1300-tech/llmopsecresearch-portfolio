#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:-root@95.133.253.123}"
REMOTE_DIM_ROOT="${2:-/root/spiral_runtime/runs/dim_narrow_main}"
LOCAL_DIM_ROOT="${3:-/mnt/d/resonance_phase16_dim_narrow_synced}"
LOCAL_CALIBRATED="${4:-/mnt/d/resonance_phase16_dim_narrow_calibrated}"
REFERENCE_ROOT="${5:-/mnt/d/resonance_phase16_boundary_outputs_v1}"
POLL_SECONDS="${POLL_SECONDS:-45}"
MAX_POLLS="${MAX_POLLS:-120}"

for ((i=1; i<=MAX_POLLS; i++)); do
  echo "[poll $i/$MAX_POLLS] pulling remote dim outputs..."
  bash /mnt/d/spiral_remote/pull_dim_narrow_results.sh \
    "$REMOTE" \
    "$REMOTE_DIM_ROOT" \
    "$LOCAL_DIM_ROOT"

  if [ -f "$LOCAL_DIM_ROOT/boundary_summary.csv" ]; then
    echo "[poll $i/$MAX_POLLS] boundary_summary.csv detected, running calibration..."
    python3 /mnt/d/resonance_dim_narrow_calibrate.py \
      --reference-root "$REFERENCE_ROOT" \
      --dim-root "$LOCAL_DIM_ROOT" \
      --output-dir "$LOCAL_CALIBRATED" \
      --reference-weight 0.60
    echo "[done] calibration complete."
    exit 0
  fi

  echo "[poll $i/$MAX_POLLS] summary not ready yet, sleeping ${POLL_SECONDS}s..."
  sleep "$POLL_SECONDS"
done

echo "[timeout] dim summary did not appear within polling window." >&2
exit 2

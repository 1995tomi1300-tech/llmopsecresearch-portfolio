#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <user@host> <remote_root>" >&2
  exit 1
fi

REMOTE="$1"
REMOTE_ROOT="$2"
KEY="${SPIRAL_SSH_KEY:-$HOME/.ssh/id_ed25519_codex_external_20260314}"

ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$REMOTE" "mkdir -p '$REMOTE_ROOT/resonance_field' '$REMOTE_ROOT/runs'"

rsync -avz --delete -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
  /mnt/d/resonance_field/ \
  "$REMOTE:$REMOTE_ROOT/resonance_field/"

rsync -avz -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
  /mnt/d/resonance_phase16_boundary.py \
  /mnt/d/run_spiral_with_telemetry.sh \
  /mnt/d/run_spiral_cpu_lane.sh \
  /mnt/d/run_spiral_phase16_boundary_lane.sh \
  "$REMOTE:$REMOTE_ROOT/"

echo "Sync complete to $REMOTE:$REMOTE_ROOT"

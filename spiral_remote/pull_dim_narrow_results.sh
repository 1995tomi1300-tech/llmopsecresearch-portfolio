#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <user@host> <remote_dim_root>" >&2
  exit 1
fi

REMOTE="$1"
REMOTE_DIM_ROOT="$2"
KEY="${SPIRAL_SSH_KEY:-$HOME/.ssh/id_ed25519_codex_external_20260314}"
LOCAL_OUT="${3:-/mnt/d/resonance_phase16_dim_narrow_synced}"

mkdir -p "$LOCAL_OUT"
rsync -avz -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
  "$REMOTE:$REMOTE_DIM_ROOT/" \
  "$LOCAL_OUT/"

echo "Pulled dim-narrow outputs to $LOCAL_OUT"

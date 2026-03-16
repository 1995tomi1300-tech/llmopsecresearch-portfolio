#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <user@host> <remote_root> [nodes-list] [variants]" >&2
  exit 1
fi

REMOTE="$1"
REMOTE_ROOT="$2"
NODES_LIST="${3:-3000,5000}"
VARIANTS="${4:-strong_balanced,critical_push}"
KEY="${SPIRAL_SSH_KEY:-$HOME/.ssh/id_ed25519_codex_external_20260314}"

ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$REMOTE" bash <<EOF
set -euo pipefail
cd "$REMOTE_ROOT"
source "$REMOTE_ROOT/.venv/bin/activate"
chmod +x "$REMOTE_ROOT/run_spiral_with_telemetry.sh" "$REMOTE_ROOT/run_spiral_cpu_lane.sh" "$REMOTE_ROOT/run_spiral_phase16_boundary_lane.sh"
tmux kill-session -t spiral-boundary 2>/dev/null || true
tmux new-session -d -s spiral-boundary \
  "SPIRAL_CPU_SET=0-29 SPIRAL_NUMERIC_THREADS=30 bash $REMOTE_ROOT/run_spiral_phase16_boundary_lane.sh $REMOTE_ROOT/runs/phase16_boundary_remote --nodes-list $NODES_LIST --variants $VARIANTS --runs 1 --steps 600 --backend gpu --max-live-gb 180"
tmux ls
EOF

echo "Remote boundary sweep started on $REMOTE"

#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <user@host> <remote_root>" >&2
  exit 1
fi

REMOTE="$1"
REMOTE_ROOT="$2"
KEY="${SPIRAL_SSH_KEY:-$HOME/.ssh/id_ed25519_codex_external_20260314}"

ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$REMOTE" "REMOTE_ROOT='$REMOTE_ROOT' bash" <<'EOF'
set -euo pipefail

source "$REMOTE_ROOT/.venv/bin/activate"

declare -a SESSION_NAMES=(
  spiral-s3000
  spiral-s5000
  spiral-s7000
  spiral-s9000
  spiral-s11000
  spiral-c3000
  spiral-c5000
  spiral-c7000
  spiral-c9000
  spiral-c11000
)

for session in "${SESSION_NAMES[@]}"; do
  tmux kill-session -t "$session" 2>/dev/null || true
done

mkdir -p \
  "$REMOTE_ROOT/runs/phase16_s3000" \
  "$REMOTE_ROOT/runs/phase16_s5000" \
  "$REMOTE_ROOT/runs/phase16_s7000" \
  "$REMOTE_ROOT/runs/phase16_s9000" \
  "$REMOTE_ROOT/runs/phase16_s11000" \
  "$REMOTE_ROOT/runs/phase16_c3000" \
  "$REMOTE_ROOT/runs/phase16_c5000" \
  "$REMOTE_ROOT/runs/phase16_c7000" \
  "$REMOTE_ROOT/runs/phase16_c9000" \
  "$REMOTE_ROOT/runs/phase16_c11000"

tmux new-session -d -s spiral-s3000 \
  "SPIRAL_CPU_SET=0-2 SPIRAL_NUMERIC_THREADS=3 bash $REMOTE_ROOT/run_spiral_phase16_boundary_lane.sh $REMOTE_ROOT/runs/phase16_s3000 --nodes-list 3000 --variants strong_balanced --runs 1 --steps 600 --backend gpu --max-live-gb 180"
tmux new-session -d -s spiral-s5000 \
  "SPIRAL_CPU_SET=3-5 SPIRAL_NUMERIC_THREADS=3 bash $REMOTE_ROOT/run_spiral_phase16_boundary_lane.sh $REMOTE_ROOT/runs/phase16_s5000 --nodes-list 5000 --variants strong_balanced --runs 1 --steps 600 --backend gpu --max-live-gb 180"
tmux new-session -d -s spiral-s7000 \
  "SPIRAL_CPU_SET=6-8 SPIRAL_NUMERIC_THREADS=3 bash $REMOTE_ROOT/run_spiral_phase16_boundary_lane.sh $REMOTE_ROOT/runs/phase16_s7000 --nodes-list 7000 --variants strong_balanced --runs 1 --steps 600 --backend gpu --max-live-gb 180"
tmux new-session -d -s spiral-s9000 \
  "SPIRAL_CPU_SET=9-11 SPIRAL_NUMERIC_THREADS=3 bash $REMOTE_ROOT/run_spiral_phase16_boundary_lane.sh $REMOTE_ROOT/runs/phase16_s9000 --nodes-list 9000 --variants strong_balanced --runs 1 --steps 600 --backend gpu --max-live-gb 180"
tmux new-session -d -s spiral-s11000 \
  "SPIRAL_CPU_SET=12-14 SPIRAL_NUMERIC_THREADS=3 bash $REMOTE_ROOT/run_spiral_phase16_boundary_lane.sh $REMOTE_ROOT/runs/phase16_s11000 --nodes-list 11000 --variants strong_balanced --runs 1 --steps 600 --backend gpu --max-live-gb 180"
tmux new-session -d -s spiral-c3000 \
  "SPIRAL_CPU_SET=15-17 SPIRAL_NUMERIC_THREADS=3 bash $REMOTE_ROOT/run_spiral_phase16_boundary_lane.sh $REMOTE_ROOT/runs/phase16_c3000 --nodes-list 3000 --variants critical_push --runs 1 --steps 600 --backend gpu --max-live-gb 180"
tmux new-session -d -s spiral-c5000 \
  "SPIRAL_CPU_SET=18-20 SPIRAL_NUMERIC_THREADS=3 bash $REMOTE_ROOT/run_spiral_phase16_boundary_lane.sh $REMOTE_ROOT/runs/phase16_c5000 --nodes-list 5000 --variants critical_push --runs 1 --steps 600 --backend gpu --max-live-gb 180"
tmux new-session -d -s spiral-c7000 \
  "SPIRAL_CPU_SET=21-23 SPIRAL_NUMERIC_THREADS=3 bash $REMOTE_ROOT/run_spiral_phase16_boundary_lane.sh $REMOTE_ROOT/runs/phase16_c7000 --nodes-list 7000 --variants critical_push --runs 1 --steps 600 --backend gpu --max-live-gb 180"
tmux new-session -d -s spiral-c9000 \
  "SPIRAL_CPU_SET=24-26 SPIRAL_NUMERIC_THREADS=3 bash $REMOTE_ROOT/run_spiral_phase16_boundary_lane.sh $REMOTE_ROOT/runs/phase16_c9000 --nodes-list 9000 --variants critical_push --runs 1 --steps 600 --backend gpu --max-live-gb 180"
tmux new-session -d -s spiral-c11000 \
  "SPIRAL_CPU_SET=27-29 SPIRAL_NUMERIC_THREADS=3 bash $REMOTE_ROOT/run_spiral_phase16_boundary_lane.sh $REMOTE_ROOT/runs/phase16_c11000 --nodes-list 11000 --variants critical_push --runs 1 --steps 600 --backend gpu --max-live-gb 180"

tmux ls
EOF

echo "B300 saturation matrix launched on $REMOTE"

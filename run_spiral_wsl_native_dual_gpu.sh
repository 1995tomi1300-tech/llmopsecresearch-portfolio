#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-modular/max-nvidia-full:latest}"
WORKSPACE="${WORKSPACE:-/mnt/d}"
GPU0_NAME="${GPU0_NAME:-spiral-native-gpu0}"
GPU1_NAME="${GPU1_NAME:-spiral-native-gpu1}"
COMMAND="${COMMAND:-mojo --version && mojo /workspace/spiral_validation.mojo}"
GPU0_COMMAND="${GPU0_COMMAND:-$COMMAND}"
GPU1_COMMAND="${GPU1_COMMAND:-$COMMAND}"
KEEP_ALIVE="${KEEP_ALIVE:-1}"
STACK_LABEL_KEY="${STACK_LABEL_KEY:-spiral.control}"
STACK_LABEL_VALUE="${STACK_LABEL_VALUE:-native-dual-gpu}"

ensure_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found in PATH" >&2
    exit 1
  fi
  docker version >/dev/null
}

cleanup_existing() {
  docker rm -f "$GPU0_NAME" "$GPU1_NAME" >/dev/null 2>&1 || true
}

start_container() {
  local gpu_index="$1"
  local name="$2"
  local inner_command="$3"
  local container_id
  if [[ "$KEEP_ALIVE" == "1" ]]; then
    inner_command="{ ${inner_command}; } ; status=\$?; echo __spiral_command_exit_code:\$status; tail -f /dev/null"
  fi
  echo "Launching $name on GPU $gpu_index"
  container_id="$(docker run --rm -d \
    --name "$name" \
    --label "${STACK_LABEL_KEY}=${STACK_LABEL_VALUE}" \
    --label "spiral.gpu_index=${gpu_index}" \
    --gpus "device=${gpu_index}" \
    -e "NVIDIA_VISIBLE_DEVICES=${gpu_index}" \
    -e "CUDA_VISIBLE_DEVICES=${gpu_index}" \
    --entrypoint /bin/bash \
    -v "${WORKSPACE}:/workspace" \
    -w /workspace \
    "$IMAGE" \
    -lc "$inner_command")"
  echo "  container_id=$container_id"
}

ensure_docker
cleanup_existing

start_container 0 "$GPU0_NAME" "$GPU0_COMMAND"
start_container 1 "$GPU1_NAME" "$GPU1_COMMAND"

cat <<EOF

Native WSL dual-GPU containers started:
  $GPU0_NAME -> GPU0
  $GPU1_NAME -> GPU1

Useful commands:
  docker logs -f $GPU0_NAME
  docker logs -f $GPU1_NAME
  docker exec -it $GPU0_NAME /bin/bash
  docker exec -it $GPU1_NAME /bin/bash

Current image: $IMAGE
Workspace mount: $WORKSPACE -> /workspace
Default command: $COMMAND
GPU0 command: $GPU0_COMMAND
GPU1 command: $GPU1_COMMAND
Keep alive: $KEEP_ALIVE
Stack label: ${STACK_LABEL_KEY}=${STACK_LABEL_VALUE}
EOF

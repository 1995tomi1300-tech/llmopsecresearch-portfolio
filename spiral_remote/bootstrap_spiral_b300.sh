#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${1:-$HOME/spiral_runtime}"
VENV_DIR="$REMOTE_ROOT/.venv"

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update
sudo apt-get install -y \
  python3 \
  python3-venv \
  python3-pip \
  git \
  rsync \
  tmux \
  htop \
  build-essential

mkdir -p "$REMOTE_ROOT"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy matplotlib

# Prefer the official CUDA-enabled PyTorch wheel. If the exact GPU/driver combo
# rejects it, we still keep the environment usable for CPU runs and can adjust.
python -m pip install \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

python - <<'PY'
import json
import platform
report = {
    "python": platform.python_version(),
}
try:
    import torch
    report["torch"] = torch.__version__
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["cuda_device_count"] = int(torch.cuda.device_count())
    if torch.cuda.is_available():
        report["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except Exception as exc:
    report["torch_error"] = str(exc)
print(json.dumps(report, indent=2))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

echo
echo "Bootstrap complete."
echo "Runtime root: $REMOTE_ROOT"
echo "Virtualenv:    $VENV_DIR"

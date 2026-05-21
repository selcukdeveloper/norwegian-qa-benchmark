# One-time environment setup on the remote workspace.

# Usage:  source h100/setup.sh

# What it does:
#   1) Installs Miniconda into ~/miniconda3
#   2) Creates a conda env "norqa" with python 3.11
#   3) Installs torch + transformers + the rest of the deps.
#   4) Exports HF cache + a couple of CUDA tunings.

# Re-running is safe — existing env / venv is reused.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

bootstrap_miniconda() {
    local target="$HOME/miniconda3"
    if [ -d "$target" ]; then
        echo "[setup] miniconda already at $target"
    else
        echo "[setup] no Python found; installing Miniconda into $target"
        local installer="$HOME/miniconda-installer.sh"
        curl -fsSL -o "$installer" \
            https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash "$installer" -b -p "$target"
        rm -f "$installer"
    fi
    source "$target/etc/profile.d/conda.sh"
    conda init bash >/dev/null
}

accept_default_channel_tos() {
    # Accept the ToS for Anaconda
    conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
    conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
}

# Python
USE_CONDA=0
if command -v conda >/dev/null 2>&1; then
    USE_CONDA=1
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -d "$HOME/miniconda3" ]; then
    # Miniconda is installed but not on PATH yet (e.g. fresh SSH session).
    USE_CONDA=1
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then
    USE_CONDA=0
else
    bootstrap_miniconda
    USE_CONDA=1
fi

if [ "$USE_CONDA" = "1" ]; then
    accept_default_channel_tos
    ENV_NAME="${ENV_NAME:-norqa}"
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        conda create -y -n "$ENV_NAME" python=3.11
    fi
    conda activate "$ENV_NAME"
    echo "[setup] using conda env '$ENV_NAME': $(python -V)"
else
    PY="$(command -v python3 || command -v python)"
    VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
    if [ ! -d "$VENV_DIR" ]; then
        echo "[setup] creating venv at $VENV_DIR with $PY"
        "$PY" -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    echo "[setup] using venv at $VENV_DIR: $(python -V)"
fi

# Install dependencies
python -m pip install --upgrade pip wheel

# Torch wheels for CUDA
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.7.0"

python -m pip install \
    "transformers==4.57.0" \
    "datasets==2.14.5" \
    "accelerate==1.10.0" \
    sentencepiece \
    bert-score \
    rouge-score \
    scikit-learn \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    tqdm

# env vars
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import torch
print(f"[setup] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[setup] gpu={torch.cuda.get_device_name(0)}")
PY

echo "[setup] done."

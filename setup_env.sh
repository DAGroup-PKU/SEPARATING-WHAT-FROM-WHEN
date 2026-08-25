#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# TCR environment setup
# conda owns the Python version; pip installs PyTorch + project packages.
# =============================================================================

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-tcr}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

echo "=== Setup Env: ${CONDA_ENV_NAME} ==="
echo "  Repo: ${REPO_ROOT}"

CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

ENV_READY=false
if conda env list | awk '{print $1}' | grep -Fx "${CONDA_ENV_NAME}" >/dev/null; then
  conda activate "${CONDA_ENV_NAME}"
  if python -c "import torch, transformers, ltx_core, tensorboard" &>/dev/null; then
    ENV_READY=true
    echo "  Conda env '${CONDA_ENV_NAME}' already ready."
    python -c "import torch; print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}')"
  fi
fi

if [ "${ENV_READY}" = true ]; then
  echo "=== Environment ready ==="
  exit 0
fi

if ! conda env list | awk '{print $1}' | grep -Fx "${CONDA_ENV_NAME}" >/dev/null; then
  echo "  Creating conda env '${CONDA_ENV_NAME}' with Python 3.12 ..."
  conda create -n "${CONDA_ENV_NAME}" python=3.12 -y
fi
conda activate "${CONDA_ENV_NAME}"
echo "  Python: $(python --version)"

if ! python -c "import torch" &>/dev/null; then
  echo "  Installing PyTorch via pip (index: ${PIP_INDEX_URL}) ..."
  pip install \
    --index-url "${PIP_INDEX_URL}" \
    --retries 5 --timeout 300 \
    torch torchvision torchaudio
fi

if ! python -c "import torch; print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}')"; then
  echo "ERROR: torch import failed."
  echo "  Try: conda remove -n ${CONDA_ENV_NAME} --all -y && bash $0"
  exit 1
fi

echo "  Installing editable project packages ..."
cd "${REPO_ROOT}"
pip install --no-deps \
  -e packages/ltx-core \
  -e packages/ltx-pipelines \
  -e packages/ltx-trainer

TORCH_VER="$(python -c 'import torch; v=torch.__version__; print(v.split("+")[0])')"
CONSTRAINTS_FILE="$(mktemp)"
cat > "${CONSTRAINTS_FILE}" <<EOF
torch==${TORCH_VER}
EOF

echo "  Installing Python deps from: ${PIP_INDEX_URL}"
pip install \
  --index-url "${PIP_INDEX_URL}" \
  -c "${CONSTRAINTS_FILE}" \
  --retries 5 --timeout 120 \
  einops \
  "transformers==4.57.6" \
  safetensors \
  "accelerate>=1.2.1" \
  "scipy>=1.14" \
  av tqdm pillow \
  "bitsandbytes>=0.45.2" \
  "huggingface-hub>=0.31.4" \
  "imageio>=2.37.0" \
  "imageio-ffmpeg>=0.6.0" \
  "opencv-python>=4.11.0.86" \
  "optimum-quanto>=0.2.6" \
  "pandas>=2.2.3" \
  "peft>=0.14.0" \
  "pillow-heif>=0.21.0" \
  "pydantic>=2.10.4" \
  "rich>=13.9.4" \
  "sentencepiece>=0.2.0" \
  "typer>=0.15.1" \
  "wandb>=0.19.11" \
  "tensorboard>=2.18" \
  "scenedetect>=0.6.5.2" \
  "pyyaml>=6.0"

pip uninstall -y torchcodec >/dev/null 2>&1 || true
rm -f "${CONSTRAINTS_FILE}"

echo "=== Environment ready ==="
python -c "import torch, ltx_core; print(f'python={__import__(\"sys\").version.split()[0]}, torch={torch.__version__}, cuda={torch.cuda.is_available()}')"

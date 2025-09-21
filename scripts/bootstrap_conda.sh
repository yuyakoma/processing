#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-kindle_ocr}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Please install Miniconda or Anaconda first." >&2
  exit 1
fi

if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists. Skipping creation." >&2
else
  echo "Creating conda environment '$ENV_NAME' with Python ${PYTHON_VERSION}..."
  conda create -y -n "$ENV_NAME" python="${PYTHON_VERSION}" tesseract poppler
fi

echo "Activating environment and installing project dependencies..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

pip install --upgrade pip
pip install -e .

echo "Environment '$ENV_NAME' is ready."

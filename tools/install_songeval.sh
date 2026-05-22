#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
SONGEVAL_DIR="$REPO_ROOT/versa_cache/SongEval"

cd "$REPO_ROOT"

"$PYTHON_BIN" -m pip install \
    "librosa==0.11.0" \
    "muq==0.1.0" \
    "hydra-core==1.3.2" \
    "safetensors"

if [ ! -d "$SONGEVAL_DIR" ]; then
    git clone https://github.com/ASLP-lab/SongEval.git "$SONGEVAL_DIR"
fi

if [ ! -f "$SONGEVAL_DIR/config.yaml" ]; then
    echo "ERROR: SongEval config.yaml is missing from $SONGEVAL_DIR."
    exit 1
fi

if [ ! -f "$SONGEVAL_DIR/ckpt/model.safetensors" ]; then
    echo "ERROR: SongEval checkpoint is missing from $SONGEVAL_DIR/ckpt."
    echo "Check that the SongEval repository checkout includes ckpt/model.safetensors."
    exit 1
fi

echo "SongEval dependencies, config, and checkpoint are ready."

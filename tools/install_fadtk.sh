#!/usr/bin/env bash

set -euo pipefail

PYTHON=${PYTHON:-python}
if [[ "${PYTHON}" == */* ]]; then
    PYTHON="$(cd "$(dirname "${PYTHON}")" && pwd)/$(basename "${PYTHON}")"
fi

run_as_root_if_needed() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        echo "Installing FFmpeg requires root privileges. Please install FFmpeg and rerun this installer." >&2
        exit 1
    fi
}

install_ffmpeg() {
    if command -v brew >/dev/null 2>&1; then
        echo "FFmpeg is required by torchcodec/FADTK; installing it with Homebrew..."
        brew install ffmpeg
    elif command -v apt-get >/dev/null 2>&1; then
        echo "FFmpeg is required by torchcodec/FADTK; installing it with apt-get..."
        run_as_root_if_needed apt-get update
        run_as_root_if_needed apt-get install -y ffmpeg
    elif command -v dnf >/dev/null 2>&1; then
        echo "FFmpeg is required by torchcodec/FADTK; installing it with dnf..."
        run_as_root_if_needed dnf install -y ffmpeg
    elif command -v yum >/dev/null 2>&1; then
        echo "FFmpeg is required by torchcodec/FADTK; installing it with yum..."
        run_as_root_if_needed yum install -y ffmpeg
    elif command -v pacman >/dev/null 2>&1; then
        echo "FFmpeg is required by torchcodec/FADTK; installing it with pacman..."
        run_as_root_if_needed pacman -Sy --noconfirm ffmpeg
    else
        echo "FFmpeg is required by torchcodec/FADTK. Please install FFmpeg and rerun this installer." >&2
        exit 1
    fi
}

if ! command -v ffmpeg >/dev/null 2>&1; then
    install_ffmpeg
fi

# NOTE(jiatong): a versa-specialized implementation for fadtk
if [ -e "fadtk" ] && [ ! -d "fadtk/.git" ]; then
    rm -rf fadtk
fi

if [ ! -d "fadtk/.git" ]; then
    git clone --depth 1 --filter=blob:none https://github.com/ftshijt/fadtk.git
fi

cd fadtk
"${PYTHON}" -m pip install -e .
"${PYTHON}" -m pip install torchcodec
cd ..

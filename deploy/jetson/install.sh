#!/usr/bin/env bash
# Jetson Orin NX setup script.
# Prerequisites: JetPack 6.x installed, Python 3.11 available.
set -euo pipefail

echo "==> Installing system dependencies…"
sudo apt-get update -q
sudo apt-get install -y \
    python3.11 python3.11-dev python3.11-venv \
    portaudio19-dev libsndfile1 \
    ffmpeg \
    git curl

echo "==> Creating virtualenv…"
python3.11 -m venv /opt/nurse-brain/venv
source /opt/nurse-brain/venv/bin/activate

echo "==> Installing Python packages…"
pip install --upgrade pip wheel

# Build llama-cpp-python with CUDA backend
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-binary llama-cpp-python

# Install the rest
pip install -e /opt/nurse-brain/[all]

echo "==> Downloading models…"
bash /opt/nurse-brain/scripts/download_models.sh

echo "==> Installing systemd service…"
sudo cp /opt/nurse-brain/deploy/jetson/systemd/nurse.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nurse.service

echo ""
echo "Installation complete. Run: sudo systemctl start nurse"

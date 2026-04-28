#!/bin/bash
# FLASH environment setup — uses venv + pip (works on Chameleon, Xavier, Pi 5, Jetson Nano)

set -e

# 1. Remove existing venv if present
if [ -d "flash" ]; then
    echo "Existing 'flash' venv found. Deleting..."
    rm -rf flash
fi

# 2. Create new venv
python3 -m venv flash

# 3. Activate
source flash/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install PyTorch — GPU build if CUDA available, otherwise CPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "CUDA detected — installing GPU PyTorch..."
    pip install torch torchvision
else
    echo "No CUDA detected — installing CPU PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# 6. Install remaining dependencies
# flwr is pinned to 1.x — Flower 2.x dropped start_server/start_numpy_client
# in favour of flower-superlink, which would require a full HFL rewrite.
pip install -r requirements.txt

echo ""
echo "Done. To activate in future sessions:"
echo "  source flash/bin/activate"

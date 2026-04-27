#!/bin/bash
# setting up the env

# 1. Install mamba if not already available
if ! command -v mamba &> /dev/null; then
    echo "Installing mamba..."
    conda install mamba -n base -c conda-forge -y
fi

# 2. Remove existing conda env if it exists
if conda env list | grep -q "^flash "; then
    echo "Existing 'flash' conda env found. Removing..."
    conda env remove -n flash -y
fi

# 3. Create new conda env with Python 3.9
mamba create -n flash python=3.9 -y

# 4. Activate
source activate flash || conda activate flash

# 5. Install PyTorch — CUDA if available, otherwise CPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing GPU PyTorch..."
    mamba install pytorch torchvision pytorch-cuda -c pytorch -c nvidia -y
else
    echo "No CUDA detected, installing CPU PyTorch..."
    mamba install pytorch torchvision cpuonly -c pytorch -y
fi

# 6. Install remaining dependencies
# Pin flwr to 1.x — Flower 2.x deprecated start_server/start_numpy_client
# in favour of flower-superlink which requires a full HFL architecture rewrite.
pip install -r requirements.txt

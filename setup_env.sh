#!/bin/bash
# setting up the env

# 1. Remove existing conda env if it exists
if conda env list | grep -q "^flash "; then
    echo "Existing 'flash' conda env found. Removing..."
    conda env remove -n flash -y
fi

# 2. Create new conda env with Python 3.9
conda create -n flash python=3.9 -y

# 3. Activate
source activate flash || conda activate flash

# 4. Install PyTorch — CUDA if available, otherwise CPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing GPU PyTorch..."
    conda install pytorch torchvision pytorch-cuda -c pytorch -c nvidia -y
else
    echo "No CUDA detected, installing CPU PyTorch..."
    conda install pytorch torchvision cpuonly -c pytorch -y
fi

# 5. Install remaining dependencies
pip install flwr psutil pynvml pandas matplotlib

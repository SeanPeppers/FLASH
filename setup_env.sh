#!/bin/bash
# setting up the env

# 1. Remove the existing environment if it exists
if [ -d "flash" ]; then
    echo "Existing 'flash' environment found. Deleting..."
    rm -rf flash
fi

# 2. Create the new virtual environment
python3 -m venv flash

# 3. Activate the environment
source flash/bin/activate

# 4. Upgrade pip and install dependencies
pip install --upgrade pip
pip install "flwr>=1.0,<2.0" torch torchvision psutil pynvml pandas matplotlib

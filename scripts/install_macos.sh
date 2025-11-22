#!/bin/bash
# Installation script for macOS (CPU-only)

echo "Installing dependencies for macOS..."

# Install PyTorch (CPU version)
pip install torch==1.5 torchvision==0.6

# Install other dependencies
pip install cython pyyaml==5.1
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install dominate==2.4.0
pip install opencv-python pillow scikit-image tqdm visdom

# Install Detectron2 (CPU version for macOS)
# Note: Detectron2 on macOS may require building from source
echo "Installing Detectron2..."
pip install 'git+https://github.com/facebookresearch/detectron2.git'

echo "Installation complete!"


#!/bin/bash
# Setup and run script for Instance-aware Image Colorization

set -e  # Exit on error

echo "=========================================="
echo "Instance-aware Image Colorization Setup"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $(python3 --version)"
echo ""

# Step 1: Install PyTorch
echo "Step 1: Installing PyTorch..."
python3 -m pip install torch torchvision
echo "✓ PyTorch installed"
echo ""

# Step 2: Install other dependencies
echo "Step 2: Installing other dependencies..."
python3 -m pip install cython pyyaml dominate opencv-python pillow scikit-image tqdm visdom
echo "✓ Dependencies installed"
echo ""

# Step 3: Install COCO API
echo "Step 3: Installing COCO API..."
python3 -m pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
echo "✓ COCO API installed"
echo ""

# Step 4: Install Detectron2
echo "Step 4: Installing Detectron2..."
echo "Note: This may take a few minutes and might require building from source..."
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git' || {
    echo "⚠ Detectron2 installation failed. You may need to build it from source."
    echo "  Try: pip install 'git+https://github.com/facebookresearch/detectron2.git'"
}
echo ""

# Step 5: Download models
echo "Step 5: Downloading pretrained models..."
if [ ! -f "checkpoints.zip" ]; then
    python3 download.py
    if [ -f "checkpoints.zip" ]; then
        unzip -q checkpoints.zip
        echo "✓ Models downloaded and extracted"
    else
        echo "⚠ Model download failed. Please download manually from:"
        echo "  https://drive.google.com/open?id=1Xb-DKAA9ibCVLqm8teKd1MWk6imjwTBh"
        echo "  Extract to the checkpoints/ folder"
    fi
else
    echo "✓ Models already downloaded"
    if [ ! -d "checkpoints/coco_finetuned_mask_256_ffs" ]; then
        unzip -q checkpoints.zip
    fi
fi
echo ""

# Step 6: Run object detection
echo "Step 6: Running object detection on example images..."
if [ ! -d "example_bbox" ]; then
    python3 inference_bbox.py --test_img_dir example
    echo "✓ Object detection complete"
else
    echo "✓ Bounding boxes already exist"
fi
echo ""

# Step 7: Run colorization
echo "Step 7: Colorizing images..."
mkdir -p results
python3 test_fusion.py --name test_fusion --sample_p 1.0 --model fusion --fineSize 256 --test_img_dir example --results_img_dir results
echo "✓ Colorization complete!"
echo ""

echo "=========================================="
echo "Setup and run complete!"
echo "Check the 'results/' folder for colorized images"
echo "=========================================="


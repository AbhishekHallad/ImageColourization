# Quick Start Guide - Run the Project

## Step-by-Step Setup (macOS)

### Option 1: Quick Setup with pip (Recommended for macOS)

1. **Install Python dependencies:**
```bash
# Install PyTorch (CPU version for macOS)
pip3 install torch torchvision

# Install other required packages
pip3 install cython pyyaml
pip3 install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install dominate opencv-python pillow scikit-image tqdm visdom

# Install Detectron2 (may take a few minutes)
pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
```

2. **Download pretrained models:**
```bash
bash scripts/download_model.sh
```

3. **Run object detection on example images:**
```bash
python inference_bbox.py --test_img_dir example
```

4. **Colorize the images:**
```bash
python test_fusion.py --name test_fusion --sample_p 1.0 --model fusion --fineSize 256 --test_img_dir example --results_img_dir results
```

5. **Check results:**
Results will be in the `results/` folder!

---

## Troubleshooting

- **If Detectron2 fails to install:** You may need to build it from source or use a different version
- **If models don't download:** Manually download from the Google Drive link in README.md
- **Memory issues:** Reduce `--fineSize` to 128 or use smaller batch sizes


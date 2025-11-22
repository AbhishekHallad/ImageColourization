# 🚀 How to Run the Project

## Quick Start (All-in-One)

Run this single command to set everything up and run the project:

```bash
./setup_and_run.sh
```

This script will:
1. ✅ Install all dependencies (PyTorch, Detectron2, etc.)
2. ✅ Download pretrained models
3. ✅ Run object detection on example images
4. ✅ Colorize the images
5. ✅ Save results to `results/` folder

---

## Manual Step-by-Step

If you prefer to run steps manually:

### 1. Install Dependencies

```bash
# Install PyTorch
pip3 install torch torchvision

# Install other packages
pip3 install cython pyyaml dominate opencv-python pillow scikit-image tqdm visdom

# Install COCO API
pip3 install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Install Detectron2 (may take a few minutes)
pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 2. Download Pretrained Models

```bash
python3 download.py
unzip checkpoints.zip
```

Or manually download from: https://drive.google.com/open?id=1Xb-DKAA9ibCVLqm8teKd1MWk6imjwTBh

### 3. Run Object Detection

```bash
python3 inference_bbox.py --test_img_dir example
```

This creates bounding boxes in `example_bbox/` folder.

### 4. Colorize Images

```bash
python3 test_fusion.py --name test_fusion --sample_p 1.0 --model fusion --fineSize 256 --test_img_dir example --results_img_dir results
```

### 5. Check Results

Look in the `results/` folder for your colorized images!

---

## Troubleshooting

### Issue: Detectron2 won't install
- **Solution**: You may need to build from source or use a Python virtual environment with Python 3.7-3.9

### Issue: Model download fails
- **Solution**: Manually download from Google Drive link above and extract to `checkpoints/`

### Issue: CUDA errors (on macOS)
- **Solution**: This is normal - the project will use CPU. Make sure you installed CPU versions of PyTorch.

### Issue: Out of memory
- **Solution**: Reduce `--fineSize` to 128 in the test command

---

## What the Project Does

1. **Object Detection**: Uses Detectron2 to find objects in images
2. **Instance Colorization**: Colorizes each detected object separately
3. **Fusion**: Combines instance-level and full-image colorization for best results

The example images in the `example/` folder will be processed and colorized results saved to `results/`.


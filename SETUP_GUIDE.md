# Setup Guide for Instance-aware Image Colorization

## Quick Start (Recommended)

### Step 1: Create Conda Environment
```bash
conda env create --file env.yml
conda activate instacolorization
```

### Step 2: Install Additional Dependencies
```bash
bash scripts/install.sh
```

**Note:** If you're on macOS or don't have CUDA, you may need to modify `scripts/install.sh`:
- For CPU-only: Remove CUDA-specific flags
- For macOS: Use CPU versions of PyTorch and Detectron2

### Step 3: Download Pretrained Models
```bash
bash scripts/download_model.sh
```

This will download the pretrained models from Google Drive and extract them to `checkpoints/`.

### Step 4: Run Object Detection on Example Images
```bash
python inference_bbox.py --test_img_dir example
```

This creates bounding boxes for all images in the `example` folder and saves them to `example_bbox/`.

### Step 5: Colorize Images
```bash
python test_fusion.py --name test_fusion --sample_p 1.0 --model fusion --fineSize 256 --test_img_dir example --results_img_dir results
```

Results will be saved in the `results/` folder.

---

## Alternative Setup (Without Conda)

If you prefer pip:

### Step 1: Install PyTorch
```bash
# For CUDA 10.1 (if you have GPU)
pip install torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html

# For CPU-only
pip install torch==1.5 torchvision==0.6
```

### Step 2: Install Other Dependencies
```bash
pip install cython pyyaml==5.1
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install dominate==2.4.0
pip install opencv-python pillow scikit-image tqdm visdom
```

### Step 3: Install Detectron2
```bash
# For CUDA 10.1
pip install detectron2==0.1.2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html

# For CPU-only (macOS/Linux)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html
```

### Step 4-5: Same as above (download models and run)

---

## Troubleshooting

### Issue: Detectron2 installation fails
- **Solution**: Try installing from source or use a newer version compatible with your PyTorch version

### Issue: CUDA errors
- **Solution**: If you don't have a GPU, make sure to install CPU-only versions of PyTorch and Detectron2

### Issue: Model download fails
- **Solution**: Manually download from [Google Drive](https://drive.google.com/open?id=1Xb-DKAA9ibCVLqm8teKd1MWk6imjwTBh) and extract to `checkpoints/`

### Issue: Out of memory
- **Solution**: Reduce `--batch_size` or `--fineSize` in the test command

---

## Testing Your Setup

After setup, test with:
```bash
python -c "import torch; import detectron2; print('Setup successful!')"
```


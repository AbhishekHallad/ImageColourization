# Colorization Pipeline Status & Recommendations

## ✅ Current Status

### What's Working Well
- **Pure full-image colorization**: No bounding boxes or masks restricting colorization
- **Adaptive AB scaling**: Per-image scaling (1.0-6.0x) based on output strength
- **Selective saturation boost**: Only pixels with existing color (magnitude > 5.0) get boosted (1.25x)
- **Luminance-aware processing**: Light midtone boost (1.1x) preserves highlight/shadow detail
- **Edge-aware smoothing**: Light Gaussian blend (10%) reduces chroma noise

### Current Results
- **Saturation range**: 0.19-0.34 (natural, balanced)
- **Color magnitude**: 9-24 (good color presence)
- **Colored pixels**: 55-96% (varies by image)
- **Lab ab range**: [-24.9, 47.1] (reasonable, no extreme values)

---

## ⚠️ Known Limitations

### Gray Faces / Low-Saturation Regions

**Why some faces stay gray:**
1. **Model uncertainty**: When the model is unsure (small faces, low detail, unusual lighting), it predicts near-zero ab values → grayscale
2. **Resolution limitation**: Current input is 256×256, which causes detail loss for small faces after network downsampling
3. **Training data**: Model trained on COCO-style dataset may not have enough close-up face examples
4. **No bounding boxes**: We removed boxes for pure colorization, but this means small faces rely entirely on global context

**Current behavior:**
- Main subjects (large, front-facing): ✅ Well colored
- Small/background faces: ⚠️ Often grayish or low-saturation
- Hands/arms/neck: ⚠️ Sometimes gray/bluish while clothes are fine

---

## 🔧 Quick Fixes (No Retraining)

### ✅ Already Implemented

1. **Selective saturation boost** (Priority 1)
   - Only boosts pixels with existing color (magnitude > 5.0)
   - Smooth transition at threshold to avoid artifacts
   - Prevents boosting near-gray areas that could create halos

2. **Adaptive scaling**
   - Per-image scaling based on output strength
   - Prevents oversaturation while ensuring visible colors

### 📋 Additional Quick Fixes (Can Implement)

1. **Face-focused second pass** (for portrait images)
   ```python
   # Run face detector on grayscale image
   # For each detected face:
   #   - Crop larger patch around face
   #   - Run colorization at higher resolution
   #   - Paste back with feather blending
   ```

2. **Increase input resolution** (if memory allows)
   ```bash
   # Test with 512×512 instead of 256×256
   python3 test_fusion.py --fineSize 512 ...
   ```

3. **Tune selective boost threshold**
   - Lower threshold (3.0-4.0) to catch more low-saturation faces
   - Increase boost factor (1.3-1.4) for stronger colors

---

## 📊 Current Pipeline

### Inference Flow
1. **Input**: Grayscale image (256×256)
2. **Model**: `netGComp` (full-image SIGGRAPH generator)
3. **Output**: Regression AB channels (normalized [-1, 1])
4. **Scaling**: Adaptive per-image (1.0-6.0x)
5. **Post-processing**:
   - Selective saturation boost (Lab space, magnitude > 5.0)
   - Light midtone boost (HSV space, 1.1x)
   - Edge-aware smoothing (Gaussian blend, 10%)

### Key Settings
- **Resolution**: 256×256 (default)
- **Sample probability**: 1.0 (no hints)
- **AB scaling**: Adaptive (0.5 + 0.05/std)
- **Selective boost**: 1.25x for colored pixels
- **Midtone boost**: 1.1x

---

## 🎯 Recommendations for Gray Faces

### Option 1: Increase Resolution (Easiest)
```bash
# Test with 512×512 (if GPU memory allows)
python3 test_fusion.py --fineSize 512 --test_img_dir example --results_img_dir results
```
**Pros**: More detail preserved, better for small faces  
**Cons**: Slower inference, more memory

### Option 2: Lower Selective Boost Threshold
Modify `color_threshold` in `fusion_model.py`:
```python
color_threshold = 3.0  # Lower from 5.0 to catch more low-saturation regions
boost_factor = 1.3     # Increase from 1.25 for stronger boost
```

### Option 3: Face-Focused Second Pass (Best for Portraits)
1. Detect faces in grayscale image
2. For each face:
   - Crop larger patch (e.g., 1.5x face size)
   - Run colorization at 512×512
   - Blend back into full image
**Pros**: Best quality for faces  
**Cons**: Requires face detection, slower

### Option 4: Fine-tune on Face Dataset (Long-term)
- Collect color face images
- Fine-tune last decoder layers
- Weight loss more on skin regions
**Pros**: Best long-term solution  
**Cons**: Requires training data and time

---

## 🔍 Debugging Checklist

- [x] ✅ No bounding boxes used (pure full-image colorization)
- [x] ✅ Adaptive scaling working (1.55x for current images)
- [x] ✅ Selective saturation boost working (64-96% colored pixels)
- [ ] ⚠️ Resolution: 256×256 (may limit small face detail)
- [ ] ⚠️ Sample_p: 1.0 (no hints, fully automatic)
- [ ] ⚠️ Model: Trained on COCO (may lack face-specific training)

---

## 📝 Next Steps

1. **Test higher resolution** (512×512) to see if it helps small faces
2. **Tune selective boost** threshold/factor for better face colors
3. **Consider face-focused second pass** for portrait images
4. **Long-term**: Fine-tune on face dataset if faces are critical

---

## Summary

The colorization pipeline is working well with natural, balanced colors. The selective saturation boost helps make existing colors pop without creating artifacts. Some gray faces are expected due to:
- Model uncertainty for small/low-detail faces
- 256×256 resolution limiting detail
- Generic training data (not face-focused)

These are normal limitations for automatic colorization. The current approach prioritizes natural-looking results over forcing color everywhere.


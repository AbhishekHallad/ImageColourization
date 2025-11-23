#!/usr/bin/env python3
"""
Diagnostic script to test colorization pipeline and verify lab2rgb conversion.
Tests synthetic color injection to verify the conversion/saving pipeline works.
"""

import numpy as np
from PIL import Image
from skimage.color import lab2rgb
import os
from skimage import io

def test_synthetic_color_injection():
    """Test 2: Create synthetic color image to verify lab2rgb pipeline works"""
    print("\n" + "="*70)
    print("TEST 2: Synthetic Color Injection (Sanity Check)")
    print("="*70)
    
    # Use one of the example images
    src = "example/e9b6826aa623549ec77bbc0275002779.jpg"
    if not os.path.exists(src):
        # Try png
        src = "example/e9b6826aa623549ec77bbc0275002779.png"
    
    if not os.path.exists(src):
        print(f"ERROR: Source image not found: {src}")
        return False
    
    print(f"Loading: {src}")
    img = Image.open(src).convert("L")  # grayscale L image
    arr = np.array(img).astype(np.float32)
    
    # Convert L from 0..255 --> 0..100 (skimage expects L in 0..100)
    L = arr / 255.0 * 100.0  # shape HxW
    
    H, W = L.shape
    print(f"Image size: {H}x{W}")
    
    # Create strong synthetic ab channels (visible green-magenta)
    a_chan = np.full((H, W), 40.0, dtype=np.float32)   # possible range -128..127
    b_chan = np.full((H, W), -20.0, dtype=np.float32)
    
    print(f"AB channels: a={a_chan[0,0]}, b={b_chan[0,0]}")
    
    lab = np.stack([L, a_chan, b_chan], axis=-1)   # H x W x 3
    
    print(f"Lab shape: {lab.shape}, L range: [{lab[:,:,0].min():.1f}, {lab[:,:,0].max():.1f}]")
    print(f"  a range: [{lab[:,:,1].min():.1f}, {lab[:,:,1].max():.1f}]")
    print(f"  b range: [{lab[:,:,2].min():.1f}, {lab[:,:,2].max():.1f}]")
    
    rgb = lab2rgb(lab)   # returns float in 0..1
    rgb8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    
    out_path = "results/test_inject_color.png"
    os.makedirs("results", exist_ok=True)
    io.imsave(out_path, rgb8)
    print(f"✅ Wrote test image: {out_path}")
    print(f"RGB stats: mean={rgb.mean():.3f}, std={rgb.std():.3f}")
    print(f"RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
    
    # Check if it's actually colored
    saturation = np.abs(rgb - rgb.mean(axis=2, keepdims=True)).mean()
    print(f"Saturation check: {saturation:.3f} (should be > 0.1 for visible color)")
    
    if saturation > 0.1:
        print("✅ SUCCESS: Synthetic color image is colored - pipeline works!")
        return True
    else:
        print("❌ WARNING: Synthetic color image appears grayscale - check lab2rgb!")
        return False


def test_forced_visible_ab():
    """Test 4: Force visible ab to test downstream pipeline"""
    print("\n" + "="*70)
    print("TEST 4: Forced Visible AB (Debug)")
    print("="*70)
    
    src = "example/e9b6826aa623549ec77bbc0275002779.jpg"
    if not os.path.exists(src):
        src = "example/e9b6826aa623549ec77bbc0275002779.png"
    
    if not os.path.exists(src):
        print(f"ERROR: Source image not found: {src}")
        return False
    
    img = Image.open(src).convert("L")
    arr = np.array(img).astype(np.float32)
    L = arr / 255.0 * 100.0
    H, W = L.shape
    
    # Create very strong visible test ab
    a_vis = np.full((H, W), 80.0, dtype=np.float32)   # strong magenta/green
    b_vis = np.full((H, W), 40.0, dtype=np.float32)
    
    print(f"Creating forced visible ab: a={a_vis[0,0]}, b={b_vis[0,0]}")
    
    lab_vis = np.stack([L, a_vis, b_vis], axis=-1)
    vis_rgb = lab2rgb(lab_vis)
    vis_rgb8 = (np.clip(vis_rgb, 0, 1) * 255).astype(np.uint8)
    
    out_path = "results/forced_visible_debug.png"
    io.imsave(out_path, vis_rgb8)
    print(f"✅ Saved: {out_path}")
    
    saturation = np.abs(vis_rgb - vis_rgb.mean(axis=2, keepdims=True)).mean()
    print(f"Saturation: {saturation:.3f} (should be > 0.2 for strong color)")
    
    if saturation > 0.2:
        print("✅ SUCCESS: Forced visible ab produces colored image!")
        return True
    else:
        print("❌ WARNING: Forced visible ab still appears grayscale!")
        return False


if __name__ == "__main__":
    print("Colorization Pipeline Diagnostics")
    print("="*70)
    
    # Test 2: Synthetic color injection
    test2_result = test_synthetic_color_injection()
    
    # Test 4: Forced visible ab
    test4_result = test_forced_visible_ab()
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print(f"  Synthetic color test: {'✅ PASS' if test2_result else '❌ FAIL'}")
    print(f"  Forced visible ab test: {'✅ PASS' if test4_result else '❌ FAIL'}")
    print("="*70)
    
    if test2_result and test4_result:
        print("\n✅ Pipeline conversion/saving works correctly!")
        print("   If model outputs are still grayscale, the issue is model output magnitude.")
    else:
        print("\n❌ Pipeline conversion/saving may have issues!")
        print("   Check lab2rgb function and saving code.")


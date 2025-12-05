"""
Quick test script to demonstrate the color editor functionality.
"""

import numpy as np
from PIL import Image
from color_editor import boost_color, recolor_region, adjust_brightness, adjust_saturation

# Create a simple test image (red, green, blue stripes)
test_img = np.zeros((256, 256, 3), dtype=np.uint8)
test_img[:, :85] = [255, 0, 0]  # Red
test_img[:, 85:170] = [0, 255, 0]  # Green
test_img[:, 170:] = [0, 0, 255]  # Blue

print("Testing color editor functions...")
print(f"Original image shape: {test_img.shape}")

# Test 1: Boost red color
print("\n1. Boosting red color by 1.3x...")
img1 = boost_color(test_img.copy(), color_name="red", factor=1.3)
print("   ✅ Red boost applied")

# Test 2: Recolor a region
print("\n2. Recoloring region (50,50) to (150,150) to blue...")
img2 = recolor_region(test_img.copy(), box=(50, 50, 150, 150), target_color_name="blue")
print("   ✅ Region recolored")

# Test 3: Adjust brightness
print("\n3. Adjusting brightness by 1.2x...")
img3 = adjust_brightness(test_img.copy(), factor=1.2)
print("   ✅ Brightness adjusted")

# Test 4: Adjust saturation
print("\n4. Adjusting saturation by 1.3x...")
img4 = adjust_saturation(test_img.copy(), factor=1.3)
print("   ✅ Saturation adjusted")

print("\n✅ All color editor functions tested successfully!")
print("\nTo use interactive editing, run:")
print("  python3 test_fusion.py --enable_editing --test_img_dir example --results_img_dir results")


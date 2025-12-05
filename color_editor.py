"""
Post-editing module for user-guided color adjustments.
Provides simple HSV/Lab-based operations for tweaking colors after automatic colorization.
"""

import cv2
import numpy as np
from PIL import Image


# Color name to hue mapping (HSV hue in degrees, OpenCV uses 0-179)
TARGET_HUES = {
    "red": 0,
    "orange": 30,
    "yellow": 60,
    "green": 120,
    "cyan": 180,
    "blue": 240,
    "purple": 280,
    "pink": 330,
}


def boost_color(image_rgb, color_name="red", factor=1.2, box=None):
    """
    Boost the saturation and brightness of a specific color (global or in a region).
    
    Args:
        image_rgb: Input RGB image (numpy array, uint8)
        color_name: Color to boost (e.g., "red", "blue", "green")
        factor: Boost factor (1.0 = no change, 1.3 = 30% brighter/more saturated)
        box: Optional tuple (x1, y1, x2, y2) to limit edit to a region, None for global
    
    Returns:
        Modified RGB image
    """
    img = image_rgb.copy()
    h, w = img.shape[:2]
    
    if box is not None:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi = img[y1:y2, x1:x2]
    else:
        roi = img
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    
    # Get target hue range for the color
    target_hue = TARGET_HUES.get(color_name.lower(), 0)
    
    # Create mask for the target color
    # OpenCV HSV: H is 0-179, so we need to handle wraparound for red (0 and 180)
    if color_name.lower() == "red":
        # Red spans both ends of hue circle
        color_mask = ((H < 15) | (H > 165)) & (S > 40)
    else:
        # For other colors, use a range around the target hue
        hue_range = 15  # degrees tolerance
        hue_min = (target_hue - hue_range) % 180
        hue_max = (target_hue + hue_range) % 180
        if hue_min < hue_max:
            color_mask = (H >= hue_min) & (H <= hue_max) & (S > 40)
        else:
            # Handle wraparound
            color_mask = ((H >= hue_min) | (H <= hue_max)) & (S > 40)
    
    # Boost saturation and brightness for matching pixels
    S[color_mask] = np.clip(S[color_mask] * factor, 0, 255)
    V[color_mask] = np.clip(V[color_mask] * factor, 0, 255)
    
    hsv[..., 1], hsv[..., 2] = S, V
    roi_out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    if box is not None:
        img[y1:y2, x1:x2] = roi_out
        return img
    else:
        return roi_out


def recolor_region(image_rgb, box, target_color_name, preserve_saturation=True):
    """
    Change the color of a region to a target color.
    
    Args:
        image_rgb: Input RGB image (numpy array, uint8)
        box: Tuple (x1, y1, x2, y2) defining the region to recolor
        target_color_name: Target color name (e.g., "blue", "red", "green")
        preserve_saturation: If True, keep original saturation; if False, use full saturation
    
    Returns:
        Modified RGB image
    """
    img = image_rgb.copy()
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    roi = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    
    # Only recolor reasonably saturated pixels (avoid grays)
    mask = S > 40
    
    # Get target hue
    target_hue_deg = TARGET_HUES.get(target_color_name.lower(), 0)
    # OpenCV hue is [0, 179], so divide by 2
    target_h = (target_hue_deg / 2.0) % 180
    
    # Change hue for masked pixels
    H[mask] = target_h
    
    # Optionally boost saturation for more vibrant colors
    if not preserve_saturation:
        S[mask] = np.clip(S[mask] * 1.2, 0, 255)
    
    hsv[..., 0] = H
    if not preserve_saturation:
        hsv[..., 1] = S
    
    roi_out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    img[y1:y2, x1:x2] = roi_out
    return img


def adjust_brightness(image_rgb, factor=1.1, box=None):
    """
    Adjust overall brightness of the image or a region.
    
    Args:
        image_rgb: Input RGB image (numpy array, uint8)
        factor: Brightness factor (1.0 = no change, 1.2 = 20% brighter, 0.8 = 20% darker)
        box: Optional tuple (x1, y1, x2, y2) to limit edit to a region, None for global
    
    Returns:
        Modified RGB image
    """
    img = image_rgb.copy()
    h, w = img.shape[:2]
    
    if box is not None:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi = img[y1:y2, x1:x2]
    else:
        roi = img
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV).astype(np.float32)
    V = hsv[..., 2]
    V = np.clip(V * factor, 0, 255)
    hsv[..., 2] = V
    
    roi_out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    if box is not None:
        img[y1:y2, x1:x2] = roi_out
        return img
    else:
        return roi_out


def adjust_saturation(image_rgb, factor=1.2, box=None):
    """
    Adjust overall saturation of the image or a region.
    
    Args:
        image_rgb: Input RGB image (numpy array, uint8)
        factor: Saturation factor (1.0 = no change, 1.3 = 30% more saturated, 0.7 = 30% less saturated)
        box: Optional tuple (x1, y1, x2, y2) to limit edit to a region, None for global
    
    Returns:
        Modified RGB image
    """
    img = image_rgb.copy()
    h, w = img.shape[:2]
    
    if box is not None:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi = img[y1:y2, x1:x2]
    else:
        roi = img
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV).astype(np.float32)
    S = hsv[..., 1]
    S = np.clip(S * factor, 0, 255)
    hsv[..., 1] = S
    
    roi_out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    if box is not None:
        img[y1:y2, x1:x2] = roi_out
        return img
    else:
        return roi_out


def interactive_edit(image_rgb, image_path):
    """
    Interactive CLI for editing a colorized image.
    
    Args:
        image_rgb: Input RGB image (numpy array, uint8)
        image_path: Path to save the edited image
    
    Returns:
        Edited RGB image
    """
    img = image_rgb.copy()
    h, w = img.shape[:2]
    
    print(f"\n📝 Post-editing options for: {image_path}")
    print(f"   Image size: {w}x{h}")
    print("\nAvailable edits:")
    print("  1. boost_color - Boost a specific color (e.g., make reds brighter)")
    print("  2. recolor - Change color of a region")
    print("  3. brightness - Adjust overall brightness")
    print("  4. saturation - Adjust overall saturation")
    print("  5. done - Finish editing")
    
    while True:
        try:
            choice = input("\nEdit choice [1-5 or 'done']: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # Non-interactive mode or user cancelled
            print("\n⚠️  Skipping editing (non-interactive mode or cancelled).")
            break
        
        if choice in ['5', 'done', '']:
            break
        
        elif choice == '1':
            try:
                print("\nAvailable colors:", ", ".join(TARGET_HUES.keys()))
                color = input("Color to boost: ").strip()
                if color.lower() not in TARGET_HUES:
                    print(f"Unknown color. Using 'red'.")
                    color = "red"
                
                try:
                    factor = float(input("Boost factor (e.g., 1.2 for 20% brighter): "))
                except ValueError:
                    factor = 1.2
                
                region = input("Apply to region? [y/n, default=n]: ").strip().lower()
                box = None
                if region == 'y':
                    try:
                        x1 = int(input("x1: "))
                        y1 = int(input("y1: "))
                        x2 = int(input("x2: "))
                        y2 = int(input("y2: "))
                        box = (x1, y1, x2, y2)
                    except ValueError:
                        print("Invalid coordinates, applying globally.")
                
                img = boost_color(img, color_name=color, factor=factor, box=box)
                print(f"✅ Boosted {color} by {factor}x")
            except (EOFError, KeyboardInterrupt):
                print("\n⚠️  Cancelled. Skipping this edit.")
                continue
        
        elif choice == '2':
            try:
                print("\nAvailable colors:", ", ".join(TARGET_HUES.keys()))
                color = input("Target color: ").strip()
                if color.lower() not in TARGET_HUES:
                    print(f"Unknown color. Using 'blue'.")
                    color = "blue"
                
                try:
                    x1 = int(input("x1: "))
                    y1 = int(input("y1: "))
                    x2 = int(input("x2: "))
                    y2 = int(input("y2: "))
                    box = (x1, y1, x2, y2)
                except ValueError:
                    print("Invalid coordinates, skipping recolor.")
                    continue
                
                preserve = input("Preserve original saturation? [y/n, default=y]: ").strip().lower()
                preserve_sat = preserve != 'n'
                
                img = recolor_region(img, box, color, preserve_saturation=preserve_sat)
                print(f"✅ Recolored region to {color}")
            except (EOFError, KeyboardInterrupt):
                print("\n⚠️  Cancelled. Skipping this edit.")
                continue
        
        elif choice == '3':
            try:
                try:
                    factor = float(input("Brightness factor (1.0=no change, 1.2=brighter, 0.8=darker): "))
                except ValueError:
                    factor = 1.1
                
                region = input("Apply to region? [y/n, default=n]: ").strip().lower()
                box = None
                if region == 'y':
                    try:
                        x1 = int(input("x1: "))
                        y1 = int(input("y1: "))
                        x2 = int(input("x2: "))
                        y2 = int(input("y2: "))
                        box = (x1, y1, x2, y2)
                    except ValueError:
                        print("Invalid coordinates, applying globally.")
                
                img = adjust_brightness(img, factor=factor, box=box)
                print(f"✅ Adjusted brightness by {factor}x")
            except (EOFError, KeyboardInterrupt):
                print("\n⚠️  Cancelled. Skipping this edit.")
                continue
        
        elif choice == '4':
            try:
                try:
                    factor = float(input("Saturation factor (1.0=no change, 1.3=more saturated, 0.7=less): "))
                except ValueError:
                    factor = 1.2
                
                region = input("Apply to region? [y/n, default=n]: ").strip().lower()
                box = None
                if region == 'y':
                    try:
                        x1 = int(input("x1: "))
                        y1 = int(input("y1: "))
                        x2 = int(input("x2: "))
                        y2 = int(input("y2: "))
                        box = (x1, y1, x2, y2)
                    except ValueError:
                        print("Invalid coordinates, applying globally.")
                
                img = adjust_saturation(img, factor=factor, box=box)
                print(f"✅ Adjusted saturation by {factor}x")
            except (EOFError, KeyboardInterrupt):
                print("\n⚠️  Cancelled. Skipping this edit.")
                continue
        
        else:
            print("Invalid choice. Please enter 1-5 or 'done'.")
    
    return img


"""
Focused crater detection - prioritize prominent central craters
"""

import cv2
import numpy as np
import os
from crater_detection import CraterDetector, visualize_detections

def detect_prominent_craters(image_path, output_path='prominent_craters.jpg'):
    """
    Focus on the most obvious, prominent craters
    Lower thresholds to catch central features
    """
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return []

    print(f"Loading: {image_path}")
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"Image shape: {img.shape}")

    # Enhanced preprocessing for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)

    # Additional preprocessing to enhance crater rims
    kernel = np.ones((3,3), np.uint8)
    img_morph = cv2.morphologyEx(img_enhanced, cv2.MORPH_CLOSE, kernel)

    # Edge detection
    edges = cv2.Canny(img_morph, 30, 100)

    # Hough circles with parameters tuned for prominent craters
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.5,         # Lower resolution to find larger circles
        minDist=150,    # Allow closer circles
        param1=80,      # Lower Canny threshold
        param2=25,      # Lower accumulator threshold (more sensitive)
        minRadius=30,   # Smaller minimum radius
        maxRadius=200   # Larger maximum radius for big central craters
    )

    craters = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        height, width = img_gray.shape

        print(f"Hough circles found: {len(circles[0])}")

        for circle in circles[0]:
            x, y, r = circle

            # Validate circle is within bounds
            if x - r >= 0 and x + r < width and y - r >= 0 and y + r < height:
                # Brightness validation for crater characteristics
                center_region = img_gray[max(0, y-10):min(height, y+10),
                                       max(0, x-10):min(width, x+10)]
                if center_region.size > 0:
                    center_brightness = np.mean(center_region)

                    # Sample rim region (outside the crater)
                    rim_inner = int(r * 0.8)  # Inner rim boundary
                    rim_outer = int(r * 1.2)  # Outer rim boundary

                    # Create circular mask for rim sampling
                    y_coords, x_coords = np.ogrid[:height, :width]
                    dist_from_center = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
                    rim_mask = (dist_from_center >= rim_inner) & (dist_from_center <= rim_outer)
                    rim_pixels = img_gray[rim_mask]

                    if len(rim_pixels) > 10:
                        rim_brightness = np.mean(rim_pixels)
                        brightness_ratio = center_brightness / max(rim_brightness, 1)

                        # Craters have dark centers relative to surroundings
                        if brightness_ratio < 0.9:  # Relaxed threshold
                            # Higher confidence for larger, more central craters
                            size_bonus = min(r / 100, 1.0)  # Larger craters get bonus
                            darkness_bonus = max(0, (0.9 - brightness_ratio) * 5)  # Darker centers get bonus

                            confidence = 0.6 + size_bonus * 0.2 + darkness_bonus * 0.2
                            confidence = min(confidence, 1.0)

                            area_fraction = (np.pi * r**2) / (height * width)

                            from crater_detection import CraterFeature
                            craters.append(CraterFeature(
                                center_x=int(x),
                                center_y=int(y),
                                radius=int(r),
                                confidence=confidence,
                                area_fraction=float(area_fraction)
                            ))

    # Sort by confidence and take top candidates
    craters.sort(key=lambda c: c.confidence, reverse=True)
    craters = craters[:20]  # Allow more detections

    print(f"\n{'='*60}")
    print(f"PROMINENT CRATERS DETECTED: {len(craters)}")
    print(f"{'='*60}\n")

    for i, crater in enumerate(craters, 1):
        print(f"Crater {i}:")
        print(f"  Location: ({crater.center_x}, {crater.center_y})")
        print(f"  Radius: {crater.radius} pixels")
        print(f"  Confidence: {crater.confidence:.3f}")
        print(f"  Size bonus: {min(crater.radius / 100, 1.0):.2f}")
        print()

    # Save visualization
    img_vis = visualize_detections(img, craters, output_path=output_path)
    print(f"Visualization saved to: {output_path}")

    return craters

if __name__ == '__main__':
    # Test on the Mars image
    image_path = 'Pictures/22574_PIA23304.jpg'
    detect_prominent_craters(image_path)
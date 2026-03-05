"""
Balanced crater detection for Mars images
Combines Hough circles with contour filtering for better accuracy
"""

import cv2
import numpy as np
import os
from crater_detection import CraterDetector, visualize_detections

def detect_craters_balanced(image_path, output_path='balanced_craters.jpg'):
    """
    Balanced approach: Hough circles + contour validation
    """
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return []

    print(f"Loading: {image_path}")
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"Image shape: {img.shape}")

    # Enhanced preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(12, 12))
    img_enhanced = clahe.apply(img_gray)

    # Gaussian blur to reduce noise
    img_blurred = cv2.GaussianBlur(img_enhanced, (3, 3), 0)

    # Edge detection
    edges = cv2.Canny(img_blurred, 40, 120)

    # Hough circles with parameters tuned for prominent craters (more sensitive)
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,          # Resolution scale
        minDist=60,      # Minimum distance between centers (reduced)
        param1=80,       # Upper threshold for Canny (reduced)
        param2=25,       # Accumulator threshold (reduced for more sensitivity)
        minRadius=15,    # Minimum radius (reduced)
        maxRadius=200    # Maximum radius (increased for large central craters)
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
                # Additional validation: check if center is relatively dark (crater floor)
                center_region = img_gray[y-5:y+5, x-5:x+5]
                if center_region.size > 0:
                    center_brightness = np.mean(center_region)
                    rim_region = img_gray[y-r:y-r+10, x-r:x+r]  # Sample rim
                    if rim_region.size > 0:
                        rim_brightness = np.mean(rim_region)
                        brightness_ratio = center_brightness / max(rim_brightness, 1)

                        # Craters typically have dark centers (relaxed threshold)
                        if brightness_ratio < 0.95:
                            confidence = 0.7 + (0.3 * (1 - brightness_ratio))
                            area_fraction = (np.pi * r**2) / (height * width)

                            from crater_detection import CraterFeature
                            craters.append(CraterFeature(
                                center_x=int(x),
                                center_y=int(y),
                                radius=int(r),
                                confidence=min(confidence, 1.0),
                                area_fraction=float(area_fraction)
                            ))

    # Sort by confidence and take top candidates
    craters.sort(key=lambda c: c.confidence, reverse=True)
    craters = craters[:25]  # Allow more detections (was 15)

    print(f"\n{'='*60}")
    print(f"FINAL CRATERS DETECTED: {len(craters)}")
    print(f"{'='*60}\n")

    for i, crater in enumerate(craters, 1):
        print(f"Crater {i}:")
        print(f"  Location: ({crater.center_x}, {crater.center_y})")
        print(f"  Radius: {crater.radius} pixels")
        print(f"  Confidence: {crater.confidence:.3f}")
        print()

    # Save visualization
    img_vis = visualize_detections(img, craters, output_path=output_path)
    print(f"Visualization saved to: {output_path}")

    return craters

if __name__ == '__main__':
    # Test on the Mars image
    image_path = 'Pictures/22574_PIA23304.jpg'
    detect_craters_balanced(image_path)
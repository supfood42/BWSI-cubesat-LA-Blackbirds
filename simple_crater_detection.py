"""
Simple crater detection for Mars images
Focuses on impact crater characteristics: circular, dark centers, bright rims
"""

import cv2
import numpy as np
import os
from crater_detection import CraterDetector, visualize_detections

def detect_mars_craters(image_path, output_path='mars_craters.jpg'):
    """
    Specialized crater detection for Mars surface images
    """
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return []

    print(f"Loading: {image_path}")
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"Image shape: {img.shape}")

    # Mars-specific preprocessing
    # 1. CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    img_enhanced = clahe.apply(img_gray)

    # 2. Morphological operations to enhance crater rims
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_morph = cv2.morphologyEx(img_enhanced, cv2.MORPH_CLOSE, kernel)

    # 3. Edge detection
    edges = cv2.Canny(img_morph, 30, 100)

    # 4. Dilate edges slightly to connect crater rims
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours (potential crater candidates)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Found {len(contours)} contour candidates")

    # Filter contours by crater-like properties
    crater_candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100 or area > 50000:  # Size filter
            continue

        # Get bounding circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)

        if radius < 15 or radius > 150:  # Radius filter
            continue

        # Calculate circularity
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        if circularity < 0.7:  # Must be fairly circular
            continue

        # Check if center is within image bounds
        height, width = img_gray.shape
        if not (radius < x < width - radius and radius < y < height - radius):
            continue

        # Calculate confidence based on circularity and size
        confidence = min(circularity * 0.8 + (1 - abs(radius - 50) / 100) * 0.2, 1.0)

        if confidence > 0.6:  # Final confidence filter
            crater_candidates.append({
                'x': int(x), 'y': int(y), 'radius': radius,
                'confidence': confidence, 'circularity': circularity
            })

    print(f"Filtered to {len(crater_candidates)} crater candidates")

    # Convert to CraterFeature format
    from crater_detection import CraterFeature
    craters = []
    height, width = img_gray.shape

    for candidate in crater_candidates:
        area_fraction = (np.pi * candidate['radius']**2) / (height * width)
        craters.append(CraterFeature(
            center_x=candidate['x'],
            center_y=candidate['y'],
            radius=candidate['radius'],
            confidence=candidate['confidence'],
            area_fraction=area_fraction
        ))

    # Sort by confidence
    craters.sort(key=lambda c: c.confidence, reverse=True)

    # Take top 10 most confident
    craters = craters[:10]

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
    detect_mars_craters(image_path)
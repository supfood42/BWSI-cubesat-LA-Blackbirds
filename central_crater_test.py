"""
Quick test for central crater detection
Focus on larger craters that might be missed
"""

import cv2
import numpy as np
import os

# Load image
image_path = 'Pictures/22574_PIA23304.jpg'
img = cv2.imread(image_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Image loaded: {img.shape}")

# Enhanced preprocessing
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
img_enhanced = clahe.apply(img_gray)

# Look specifically for larger craters
edges = cv2.Canny(img_enhanced, 30, 100)

# Multiple Hough circle attempts with different parameters
print("Testing different parameter sets...")

results = []

# Test 1: Large craters
circles1 = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=2, minDist=300,
                           param1=80, param2=20, minRadius=100, maxRadius=400)
if circles1 is not None:
    results.append(("Large craters", len(circles1[0]), circles1))
    print(f"Large craters: {len(circles1[0])} found")

# Test 2: Medium-large craters
circles2 = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.5, minDist=200,
                           param1=80, param2=25, minRadius=80, maxRadius=300)
if circles2 is not None:
    results.append(("Medium-large", len(circles2[0]), circles2))
    print(f"Medium-large: {len(circles2[0])} found")

# Test 3: Central area focus
height, width = img_gray.shape
center_y, center_x = height // 2, width // 2
roi = img_enhanced[center_y-400:center_y+400, center_x-400:center_x+400]
if roi.size > 0:
    roi_edges = cv2.Canny(roi, 30, 100)
    circles3 = cv2.HoughCircles(roi_edges, cv2.HOUGH_GRADIENT, dp=1.5, minDist=150,
                               param1=80, param2=20, minRadius=50, maxRadius=200)
    if circles3 is not None:
        # Adjust coordinates back to full image
        circles3[0, :, 0] += center_x - 400  # x offset
        circles3[0, :, 1] += center_y - 400  # y offset
        results.append(("Central ROI", len(circles3[0]), circles3))
        print(f"Central ROI: {len(circles3[0])} found")

# Combine and visualize all results
img_combined = img.copy()
total_craters = 0

colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Red, Blue

for i, (label, count, circles) in enumerate(results):
    color = colors[i % len(colors)]
    circles_int = np.uint16(np.around(circles))

    for circle in circles_int[0, :]:
        x, y, r = circle
        if 0 <= x < width and 0 <= y < height and r > 0:
            cv2.circle(img_combined, (x, y), r, color, 3)
            cv2.circle(img_combined, (x, y), 2, color, -1)
            total_craters += 1

cv2.imwrite('central_crater_test.jpg', img_combined)
print(f"\nTotal craters found: {total_craters}")
print("Combined result saved as: central_crater_test.jpg")

# Show what we found
for label, count, circles in results:
    print(f"{label}: {count} craters")
    if count > 0 and count <= 5:  # Show details for small numbers
        circles_int = np.uint16(np.around(circles))
        for j, circle in enumerate(circles_int[0, :min(3, count)]):
            x, y, r = circle
            print(f"  {j+1}: center=({x}, {y}), radius={r}")
"""
CubeSat Image Processing Pipeline
Processes captured imagery to detect geological features (craters).
Optimized for onboard processing with limited computational resources.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from crater_detection import CraterDetector, visualize_detections


class ImageProcessor:
    """Processes satellite imagery for feature detection."""
    
    def __init__(self):
        self.crater_detector = CraterDetector(confidence_threshold=0.3)
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        return img
    
    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocessing pipeline: grayscale, enhancement, filtering.
        
        Parameters:
            img_bgr: BGR image from OpenCV
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # CLAHE enhancement (Contrast Limited Adaptive Histogram Equalization)
        # Useful for bringing out crater edges without over-amplifying noise
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_gray)
        
        # Noise reduction via Gaussian blur
        img_filtered = cv2.GaussianBlur(img_enhanced, (5, 5), 0)
        
        return img_filtered
    
    def process_image(self, image_path: str, detect_craters: bool = True) -> dict:
        """
        Full image processing pipeline.
        
        Parameters:
            image_path: path to image file
            detect_craters: whether to run crater detection
            
        Returns:
            Dictionary with processed image data and detections
        """
        # Load
        img_bgr = self.load_image(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        img_processed = self.preprocess(img_bgr)
        
        # Detect craters
        craters = []
        if detect_craters:
            craters = self.crater_detector.detect_craters(img_processed, method='hough')
        
        result = {
            'image_path': image_path,
            'image_shape': img_bgr.shape,
            'craters_detected': len(craters),
            'features': [crater.to_dict() for crater in craters],
            'processed_image': img_processed,
            'visual_image': img_rgb,
            'crater_objects': craters
        }
        
        return result


def main():
    """Interactive image processing pipeline."""
    print("CubeSat Image Processing Pipeline")
    print("=" * 50)
    
    # Select image
    root = Tk()
    root.withdraw()
    image_path = askopenfilename(
        title="Select an image file",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
    )
    root.destroy()
    
    if not image_path:
        print("No image selected.")
        return
    
    # Process
    processor = ImageProcessor()
    result = processor.process_image(image_path, detect_craters=True)
    
    print(f"Image loaded: {os.path.basename(image_path)}")
    print(f"Image shape: {result['image_shape']}")
    print(f"Craters detected: {result['craters_detected']}")
    
    # Display original
    plt.figure(figsize=(6, 6))
    plt.imshow(result['visual_image'])
    plt.title("Original Image")
    plt.axis("off")
    plt.show()
    
    # Display processed
    plt.figure(figsize=(6, 6))
    plt.imshow(result['processed_image'], cmap='gray')
    plt.title("Processed Image (CLAHE Enhanced)")
    plt.axis("off")
    plt.show()
    
    # Display with detections
    img_with_craters = visualize_detections(result['visual_image'], result['crater_objects'])
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img_with_craters, cv2.COLOR_BGR2RGB))
    plt.title(f"Crater Detections ({result['craters_detected']} found)")
    plt.axis("off")
    plt.show()
    
    # Print feature metadata
    if result['features']:
        print("\nDetected Features:")
        for i, feature in enumerate(result['features'], 1):
            print(f"  Crater {i}: center={feature['center']}, radius={feature['radius_px']}px, "
                  f"confidence={feature['confidence']}, area_fraction={feature['area_fraction']}")
    
    # Save metadata
    metadata_path = image_path.replace('.jpg', '_metadata.json').replace('.png', '_metadata.json')
    with open(metadata_path, 'w') as f:
        # Make JSON serializable
        metadata = {
            'image_path': result['image_path'],
            'image_shape': result['image_shape'],
            'craters_detected': result['craters_detected'],
            'features': result['features']
        }
        json.dump(metadata, f, indent=2)
        print(f"\nMetadata saved to: {metadata_path}")


if __name__ == '__main__':
    main()

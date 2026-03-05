"""
Crater Detection Module for CubeSat Imaging Payload
Detects circular features (craters, depressions) in surface imagery
using Hough circle transform and contour analysis.

No clue if ts work or not btw
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CraterFeature:
    """Metadata for detected crater feature."""
    center_x: int          # pixel coordinate
    center_y: int          # pixel coordinate
    radius: int            # pixels
    confidence: float      # 0.0 to 1.0
    area_fraction: float   # fraction of image area occupied
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'center': (self.center_x, self.center_y),
            'radius_px': self.radius,
            'confidence': round(self.confidence, 3),
            'area_fraction': round(self.area_fraction, 4)
        }


class CraterDetector:
    """
    Detects circular features (craters) in preprocessed images.
    Optimized for CubeSat onboard processing.
    """
    
    def __init__(self, confidence_threshold=0.3):
        """
        Initialize crater detector.
        
        Parameters:
            confidence_threshold: minimum confidence score to report detections (0.0-1.0)
        """
        self.confidence_threshold = confidence_threshold
    
    def detect_craters_hough(self, image: np.ndarray) -> List[CraterFeature]:
        """
        Detect circular craters using Hough circle transform.
        
        Parameters:
            image: preprocessed grayscale image
            
        Returns:
            List of detected CraterFeature objects
        """
        if image is None or image.size == 0:
            return []
        
        # Normalize image if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Apply edge detection for circle finding
        edges = cv2.Canny(image, 50, 150)
        
        # Hough circle detection - more conservative parameters
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,        # Minimum distance between centers (increased)
            param1=100,        # Upper threshold for Canny
            param2=50,         # Accumulator threshold (increased for fewer false positives)
            minRadius=10,      # Minimum radius (increased)
            maxRadius=80       # Maximum radius (decreased)
        )
        
        features = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            image_height, image_width = image.shape
            image_area = image_height * image_width
            
            for circle in circles[0, :]:
                x, y, r = circle
                # Validate circle is within image bounds
                if x - r >= 0 and x + r < image_width and y - r >= 0 and y + r < image_height:
                    confidence = 0.75  # Base confidence for Hough detection
                    area_fraction = (np.pi * r**2) / image_area
                    
                    if confidence >= self.confidence_threshold:
                        features.append(CraterFeature(
                            center_x=int(x),
                            center_y=int(y),
                            radius=int(r),
                            confidence=confidence,
                            area_fraction=float(area_fraction)
                        ))
        
        return features
    
    def detect_craters_contour(self, image: np.ndarray) -> List[CraterFeature]:
        """
        Detect crater-like features using contour analysis and circularity.
        Alternative to Hough circles for different image conditions.
        
        Parameters:
            image: preprocessed grayscale image
            
        Returns:
            List of detected CraterFeature objects
        """
        if image is None or image.size == 0:
            return []
        
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Binary threshold
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        features = []
        image_height, image_width = image.shape
        image_area = image_height * image_width
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # Skip tiny contours
                continue
            
            # Fit circle to contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            if radius < 5 or radius > 100:  # Filter by realistic crater size
                continue
            
            # Calculate circularity: ratio of contour area to fitted circle area
            circle_area = np.pi * radius**2
            circularity = float(area) / circle_area if circle_area > 0 else 0
            
            # High circularity (close to 1.0) indicates crater-like shape
            if circularity >= 0.6:
                confidence = min(circularity, 1.0)
                area_fraction = circle_area / image_area
                
                if confidence >= self.confidence_threshold:
                    features.append(CraterFeature(
                        center_x=int(x),
                        center_y=int(y),
                        radius=int(radius),
                        confidence=float(confidence),
                        area_fraction=float(area_fraction)
                    ))
        
        return features
    
    def detect_craters(self, image: np.ndarray, method: str = 'hough') -> List[CraterFeature]:
        """
        Detect craters using specified method.
        
        Parameters:
            image: preprocessed grayscale image
            method: 'hough' or 'contour'
            
        Returns:
            List of detected CraterFeature objects, sorted by confidence (descending)
        """
        if method == 'contour':
            features = self.detect_craters_contour(image)
        else:
            features = self.detect_craters_hough(image)
        
        # Sort by confidence, highest first
        features.sort(key=lambda f: f.confidence, reverse=True)
        return features
    
    def filter_by_size(self, features: List[CraterFeature], 
                      min_px: int = 5, max_px: int = 100) -> List[CraterFeature]:
        """Filter detected craters by radius range in pixels."""
        return [f for f in features if min_px <= f.radius <= max_px]
    
    def filter_by_confidence(self, features: List[CraterFeature], 
                            threshold: float = 0.5) -> List[CraterFeature]:
        """Filter detected craters by confidence threshold."""
        return [f for f in features if f.confidence >= threshold]


def visualize_detections(image: np.ndarray, features: List[CraterFeature], 
                        output_path: str = None) -> np.ndarray:
    """
    Visualize detected craters on image.
    
    Parameters:
        image: original or preprocessed image
        features: list of CraterFeature objects
        output_path: optional path to save visualization
        
    Returns:
        Image with drawn circles
    """
    # Convert to RGB if grayscale for visualization
    if len(image.shape) == 2:
        img_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_vis = image.copy()
    
    # Draw detected circles
    for feature in features:
        cv2.circle(img_vis, (feature.center_x, feature.center_y), 
                  feature.radius, (0, 255, 0), 2)
        cv2.circle(img_vis, (feature.center_x, feature.center_y), 
                  3, (0, 0, 255), -1)  # Center point
    
    if output_path:
        cv2.imwrite(output_path, img_vis)
    
    return img_vis

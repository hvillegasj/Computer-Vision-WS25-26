"""
Task 2: Hough Transform for Circle Detection
Task 3: Mean Shift for Peak Detection in Hough Accumulator
Template for MA-INF 2201 Computer Vision WS25/26
Exercise 03
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os


def myHoughCircles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz):
    """
    Your implementation of HoughCircles
    
    Args:
        edges: single-channel binary source image (e.g: edges)
        min_radius: minimum circle radius
        max_radius: maximum circle radius
        param threshold: minimum number of votes to consider a detection
        min_dist: minimum distance between two centers of the detected circles. 
        r_ssz: stepsize of r
        theta_ssz: stepsize of theta
        return: list of detected circles as (a, b, r, v), accumulator as [r, y_c, x_c]
    """
    max_radius = min(max_radius, int(np.linalg.norm(edges.shape)))

    edges_points = np.array(np.nonzero(edges))
    h, w = edges.shape

    # TODO

    # return detected_circles, accumulator

def myMeanShift(accumulator, bandwidth, threshold=None):
    """
    Find peaks in Hough accumulator using mean shift.
    
    Args:
        accumulator: 3D Hough accumulator (n_radii, h, w)
        bandwidth: Bandwidth for mean shift
        threshold: Minimum value to consider (if None, use fraction of max)
        
    Returns:
        peaks: List of (x, y, r_idx, value) tuples
    """
    n_r, h, w = accumulator.shape
    
    # TODO
    
    # return peaks

def main():
    
    print("=" * 70)
    print("Task 2: Hough Transform for Circle Detection")
    print("=" * 70)
        
    img_path = 'data/coins.jpg'
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        return
        
    # Load image and convert to grayscale
    # TODO
    
    # Apply Canny edge detection
    # TODO
    
    # Detect circles - parameters tuned for coins image
    print("\nDetecting circles...")
    # min_radius = 
    # max_radius = 
    # threshold = 
    # min_dist = 
    # r_ssz = 
    # theta_ssz = 
    
    # TODO
    # detected_circles, accumulator = myHoughCircles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz)

    # Visualize detected circles
    # TODO
    
    # Visualize accumulator slices
    # TODO
    
    # Visualize peak radius
    # TODO
    
    print("\n" + "=" * 70)
    print("Parameter Analysis:")
    print("  - Canny thresholds affect edge quality and thus detection")
    # ...more analysis can be added here
    print("=" * 70)
    print("Task 2 complete!")
    print("=" * 70)


    # =============================================================
    print("=" * 70)
    print("Task 3: Mean Shift for Peak Detection in Hough Accumulator")
    print("=" * 70)

    print("Applying mean shift to find peaks...")
    # peaks = myMeanShift # TODO
    
    # Visualize corresponding circles on original image    
    # TODO
    
    print("\n" + "=" * 70)
    print("Bandwidth Parameter Analysis:")
    # ...more analysis can be added here
    print("=" * 70)
    print("Task 3 complete!")
    

if __name__ == "__main__":
    main()
"""
Task 1: Distance Transform using Chamfer 5-7-11
Template for MA-INF 2201 Computer Vision WS25/26
Exercise 03
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def chamfer_distance_transform_5_7_11(binary_image):
    """
    Compute Chamfer distance transform using 5-7-11 mask.
    
    Based on Borgefors "Distance transformations in digital images" (1986).
    
    Chamfer 5-7-11:
    - Horizontal/vertical neighbors: weight = 5
    - Diagonal neighbors: weight = 7
    - Knight's move neighbors: weight = 11
    
    Args:
        binary_image: Binary image where features are 255, background is 0
    
    Returns:
        Distance transform image
    """
    h, w = binary_image.shape
    dt = np.full((h, w), np.inf, dtype=np.float32)
    
    # Initialize: 0 if feature pixel, infinity otherwise
    dt[binary_image > 0] = 0
    
    # Define forward and backward masks with (row_offset, col_offset, distance)
    # Forward mask (as shown in slide 37)
    # TODO
    
    # Backward mask (as shown in slide 37)
    # TODO
    
    # Forward pass
    # TODO
    
    # Backward pass
    # TODO
    
    return dt


def main():    
    
    print("=" * 70)
    print("Task 1: Distance Transform using Chamfer 5-7-11")
    print("=" * 70)
    
    img_path = 'data/bonn.jpg'
    # img_path = 'data/circle.png'      # play with different images
    # img_path = 'data/square.png'      
    # img_path = 'data/triangle.png'    
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        return
    
    # Load image and convert to grayscale
    # TODO
    
    # Apply Canny edge detection
    # TODO
    
    # Compute distance transform with the function chamfer_distance_transform_5_7_11
    # TODO

    # Compute distance transform using cv2.distanceTransform
    # TODO

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # TODO
    # 1. Original image
    # 2. Edge image
    # 3. Distance transform
    # 4. Distance transform using OpenCV
        
    print("\n" + "=" * 70)
    print("Task 1 complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
    
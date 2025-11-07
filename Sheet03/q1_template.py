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
    forward_mask = [
        (-2, -1, 11), (-2, 1, 11),
        (-1, -2, 11), (-1, -1, 7), (-1, 0, 5), (-1, 1, 7), (-1, 2, 11),
        (0, -2, 11), (0, -1, 5)
    ]
    
    # Backward mask (as shown in slide 37)
    backward_mask = [
        (2, -1, 11), (2, 1, 11),
        (1, -2, 11), (1, -1, 7), (1, 0, 5), (1, 1, 7), (1, 2, 11),
        (0, 1, 5), (0, 2, 11)
    ]
    
    # Forward pass
    for i in range(h):
        for j in range(w):
            vij = dt[i, j]
            for dy, dx, wgt in forward_mask:
                y, x = i + dy, j + dx
                if 0 <= y < h and 0 <= x < w:          # boundary check
                    cand = dt[y, x] + wgt
                    if cand < vij:
                        vij = cand
            dt[i, j] = vij

    # Backward pass
    for i in range(h - 1, -1, -1):
        for j in range(w - 1, -1, -1):
            vij = dt[i, j]
            for dy, dx, wgt in backward_mask:
                y, x = i + dy, j + dx
                if 0 <= y < h and 0 <= x < w:          # boundary check
                    cand = dt[y, x] + wgt
                    if cand < vij:
                        vij = cand
            dt[i, j] = vij
                
    
    return dt


def main():    
    
    print("=" * 70)
    print("Task 1: Distance Transform using Chamfer 5-7-11")
    print("=" * 70)
    
    img_path = 'data/bonn.jpg'
    #img_path = 'data/circle.png'      # play with different images
    #img_path = 'data/square.png'      
    #img_path = 'data/triangle.png'    
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        return
    
    # Load image and convert to grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Canny edge detection
    edges = cv2.Canny(img, threshold1=50, threshold2=150, L2gradient=True)
    
    # Compute distance transform with the function chamfer_distance_transform_5_7_11
    chamfer_dt = chamfer_distance_transform_5_7_11(edges)

    # Compute distance transform using cv2.distanceTransform
    inv_edges = cv2.bitwise_not(edges)  # edges(255) -> 0, background(0) -> 255
    cv_dt = cv2.distanceTransform(inv_edges, distanceType=cv2.DIST_L2, maskSize=5)

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax = axes.ravel()
    # 1. Original image
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original (grayscale)')
    ax[0].axis('off')
    # 2. Edge image
    ax[1].imshow(edges, cmap='gray')
    ax[1].set_title('Canny edges')
    ax[1].axis('off')
    # 3. Distance transform
    im2 = ax[2].imshow(chamfer_dt / 5.0, cmap='inferno')
    ax[2].set_title('Chamfer DT (5-7-11)')
    ax[2].axis('off')
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    # 4. Distance transform using OpenCV
    im3 = ax[3].imshow(cv_dt, cmap='inferno')
    ax[3].set_title('OpenCV Distance Transform ')
    ax[3].axis('off')
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
   
    
        
    print("\n" + "=" * 70)
    print("Task 1 complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
    
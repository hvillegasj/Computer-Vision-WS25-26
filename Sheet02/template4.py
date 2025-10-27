# Template for Exercise 4 – NCC Stereo Matching

import cv2
import numpy as np
import matplotlib.pyplot as plt


WINDOW_SIZE = 11       # NCC patch size
MAX_DISPARITY = 64     # Maximum search range


def compute_manual_ncc_map(left_image, right_image, window_size, max_disparity):
    """
    Compute a dense disparity map using Normalized Cross-Correlation (NCC).
    
    Arguments:
        left_image, right_image : input grayscale stereo pair
        window_size             : size of the correlation window
        max_disparity           : maximum horizontal shift to consider

    Returns:
        disparity_map : computed disparity for each pixel (float32)
    """
    pass


def compute_mae(a, b, mask=None):
    """
    Compute Mean Absolute Error (MAE) between two disparity maps.
    Optionally, use a mask to exclude invalid pixels.
    """
    pass


# ==========================================================


# TODO: 1. Load the stereo image pair (left.png, right.png) in grayscale
# TODO: 2. Call your NCC function to compute the manual disparity map
# TODO: 3. Compute a benchmark map using cv2.StereoBM_create with the same parameters
# TODO: 4. Visualize both maps and compare them qualitatively
# TODO: 5. Quantitatively compare both maps by computing MAE (Mean Absolute Error)
# TODO: 6. Ensure your manual implementation achieves MAE < 0.7 pixels

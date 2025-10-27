# Template for Exercise 5 – Canny Edge Detector

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def gaussian_smoothing(img, sigma):
    """
    Apply Gaussian smoothing to reduce noise.
    """
    pass


def compute_gradients(img):
    """
    Compute gradient magnitude and direction (Sobel-based).
    Return gradient_magnitude, gradient_angle.
    """
    pass


def nonmax_suppression(mag, ang):
    """
    Perform non-maximum suppression to thin edges.
    """
    pass


def double_threshold(nms, low, high):
    """
    Apply double thresholding to classify strong, weak, and non-edges.
    Return thresholded edge map.
    """
    pass


def hysteresis(edge_map, weak, strong):
    """
    Perform edge tracking by hysteresis.
    Return final binary edge map.
    """
    pass


def compute_metrics(manual_edges, cv_edges):
    """
    Compute MAD, precision, recall, and F1-score between two binary edge maps.
    """
    pass


# ==========================================================

# TODO: 1. Load the grayscale image 'bonn.jpg'
# TODO: 2. Smooth the image using your Gaussian function
# TODO: 3. Compute gradients (magnitude and direction)
# TODO: 4. Apply non-maximum suppression
# TODO: 5. Apply double threshold (choose suitable low/high values)
# TODO: 6. Perform hysteresis to obtain final edges
# TODO: 7. Compare your result with cv2.Canny using MAD and F1-score
# TODO: 8. Display original image, your edges, and OpenCV edges

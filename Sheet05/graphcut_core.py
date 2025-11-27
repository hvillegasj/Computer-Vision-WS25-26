import cv2
import numpy as np
import maxflow
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


# TODO: Your implementation here

img_path = 'Sheet05\dataset\images\bike_2007_005878.jpg'
labels_path = 'Sheet05\dataset\images-labels\bike_2007_005878-anno.png'


if not os.path.exists(img_path):
    print(f"Error: {img_path} not found!")

if not os.path.exists(labels_path):
    print(f"Error: {labels_path} not found!")

img = cv2.imread(img_path)

# TODO: Smooth the image

def get_histogram(window):
    """
    Returns the histogram of the colors in the window
    Args:
        window: Window of which the histogram will be extracted.
    Output:
        histogram: Color histogram of the window.
    """
    bins = [np.arange(257), np.arange(257), np.arange(257)]

    data = window.reshape(-1, 3)
    
    histogram, _ = np.histogramdd(data, bins=bins)
    
    return histogram

def populate_training_data(origin_img, label_img, window_size):
    """
    Fills an array with the color histograms.
    Args:
        origin_img: Original image
        label_img: Label mask to design the pixels that are to be used
        window_size: Size of the window to calculate the histograms upon
    Output:
        histograms_vector: Vector of the histograms of the pixels of interest
    """
    histograms_vector = []
    h, w = origin_img.shape[:2]

    non_zero_mask = (label_img != 0)
    non_zero_mask = np.any(non_zero_mask, axis = -1)
    y_coords, x_coords = np.nonzero(non_zero_mask)  

    for y, x in zip(y_coords, x_coords):
        valid_min_y = np.max([0, y - window_size])
        valid_max_y = np.min([y + window_size + 1, h])
        valid_min_x = np.max([0, x - window_size])
        valid_max_x = np.min([x + window_size + 1, w])

        histogram = get_histogram(origin_img[valid_min_y:valid_max_y, valid_min_x:valid_max_x])
        histograms_vector.append(histogram)
    
    return np.array(histograms_vector)
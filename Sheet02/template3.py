# Template for Exercise 3 – Spatial and Frequency Domain Filtering
import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_box_kernel(k):
    """
    Create a normalized k×k box filter kernel.
    """
    pass


def make_gauss_kernel(k, sigma):
    """
    Create a normalized 2D Gaussian filter kernel of size k×k.
    """
    pass


def conv2_same_zero(img, h):
    """
    Perform 2D spatial convolution using zero padding.
    Output should have the same size as the input image.
    (Do NOT use cv2.filter2D)
    """
    pass


def freq_linear_conv(img, h):
    """
    Perform linear convolution in the frequency domain.
    (You can use numpy.fft)
    """
    pass


def compute_mad(a, b):
    """
    Compute Mean Absolute Difference (MAD) between two images.
    """
    pass

# ==========================================================

# TODO: 1. Load the grayscale image (e.g., lena.png)
# TODO: 2. Construct 9×9 box and Gaussian kernels (same sigma)
# TODO: 3. Apply both filters spatially (manual convolution)
# TODO: 4. Apply both filters in the frequency domain
# TODO: 5. Compute and print MAD between spatial and frequency outputs
# TODO: 6. Visualize all results (original, box/gaussian spatial, box/gaussian frequency, spectrum)
# TODO: 7. Verify that MAD < 1×10⁻⁷ for both filters

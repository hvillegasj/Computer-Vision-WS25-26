import cv2
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt

# ==============================================================================
# 0. Setup and Image Loading
# ==============================================================================
print("--- 0. Setup: Loading Images ---")

'''
TODO: Load the original image 'bonn.jpg' and noisy image 'bonn_noisy.jpg'
Convert both to grayscale and prepare the noisy image in float format (0-1 range)
Calculate and print the PSNR of the noisy image compared to the original
'''

# Load images here
original_img_color = None  # Load bonn.jpg
original_img_gray = None   # Convert to grayscale
noisy_img = None           # Load bonn_noisy.jpg and convert to grayscale
noisy_img_float_01 = None  # Convert noisy image to float format (0-1)

# Calculate PSNR of noisy image
psnr_noisy = None

# Display original and noisy images
# TODO: Create a figure showing original and noisy images side by side


# ==============================================================================
# Custom Filter Definitions (for parts a, b, c)
# ==============================================================================

def custom_gaussian_filter(image, kernel_size, sigma):
    """
    Custom Gaussian Filter - Implement convolution from scratch
    
    Args:
        image: Input image (float, 0-1 range)
        kernel_size: Size of the Gaussian kernel (odd integer)
        sigma: Standard deviation of the Gaussian
    
    Returns:
        Filtered image (float, 0-1 range)
    
    TODO: 
    1. Create Gaussian kernel using the formula: G(x,y) = exp(-(x^2 + y^2)/(2*sigma^2))
    2. Normalize the kernel so it sums to 1
    3. Pad the image using reflect mode
    4. Apply convolution manually using nested loops
    """
    pass


def custom_median_filter(image, kernel_size):
    """
    Custom Median Filter - Implement median calculation from scratch
    
    Args:
        image: Input image (float, 0-1 range)
        kernel_size: Size of the median filter window (odd integer)
    
    Returns:
        Filtered image (float, 0-1 range)
    
    TODO:
    1. Pad the image using reflect mode
    2. For each pixel, extract the neighborhood window
    3. Calculate the median of the window
    4. Assign the median value to the output pixel
    """
    pass


def custom_bilateral_filter(image, d, sigma_color, sigma_space):
    """
    Custom Bilateral Filter
    
    Args:
        image: Input image (float, 0-1 range)
        d: Diameter of the pixel neighborhood
        sigma_color: Filter sigma in the color space (0-1 range for float images)
        sigma_space: Filter sigma in the coordinate space
    
    Returns:
        Filtered image (float, 0-1 range)
    
    TODO:
    1. Pad the image
    2. For each pixel:
       a. Calculate spatial weights based on distance from center
       b. Calculate range weights based on intensity difference
       c. Combine weights and compute weighted average
    3. Normalize by sum of weights
    """
    pass


# ==============================================================================
# 1. Filter Application (Parts a, b, c)
# ==============================================================================
print("\n--- 1. Filter Application (Parts a, b, c) ---")

# Default Parameters
K_DEFAULT = 7
S_DEFAULT = 2.0
D_DEFAULT = 9
SC_DEFAULT = 100  # cv2 range (0-255)
SS_DEFAULT = 75

# -------------------------- a) Gaussian Filter --------------------------
print("a) Applying Gaussian Filter...")
'''
TODO: 
1. Apply Gaussian filter using cv2.GaussianBlur()
2. Apply your custom Gaussian filter
3. Calculate PSNR for both results
4. Display the results in a figure with 3 subplots (noisy, cv2 result, custom result)
'''

denoised_gaussian_cv2 = None
psnr_gaussian_cv2 = None

denoised_gaussian_custom = None
psnr_gaussian_custom = None

# Display results here


# -------------------------- b) Median Filter --------------------------
print("b) Applying Median Filter...")
'''
TODO:
1. Apply Median filter using cv2.medianBlur()
2. Apply your custom Median filter
3. Calculate PSNR for both results
4. Display the results in a figure with 3 subplots
'''

denoised_median_cv2 = None
psnr_median_cv2 = None

denoised_median_custom = None
psnr_median_custom = None

# Display results here


# -------------------------- c) Bilateral Filter --------------------------
print("c) Applying Bilateral Filter...")
'''
TODO:
1. Apply Bilateral filter using cv2.bilateralFilter()
2. Apply your custom Bilateral filter (remember to scale sigma_color for 0-1 range)
3. Calculate PSNR for both results
4. Display the results in a figure with 3 subplots
'''

denoised_bilateral_cv2 = None
psnr_bilateral_cv2 = None

denoised_bilateral_custom = None
psnr_bilateral_custom = None

# Display results here


# ==============================================================================
# 2. Performance Comparison (Part d)
# ==============================================================================
print("\n--- d) Performance Comparison ---")
'''
TODO:
1. Compare PSNR values of all three filters
2. Determine which filter performs best
3. Display side-by-side comparison of all filtered images
4. Print the results with the best performing filter highlighted
'''


# ==============================================================================
# 3. Parameter Optimization (Part e)
# ==============================================================================

def run_optimization(original_img, noisy_img):
    """
    Optimize parameters for all three filters to maximize PSNR
    
    Args:
        original_img: Original clean image
        noisy_img: Noisy image to be filtered
    
    Returns:
        Dictionary containing optimal parameters and best PSNR for each filter
    
    TODO:
    1. For Gaussian filter: iterate over kernel_sizes and sigma values
    2. For Median filter: iterate over kernel_sizes
    3. For Bilateral filter: iterate over d, sigma_color, and sigma_space values
    4. Track the best PSNR and corresponding parameters for each filter
    5. Return results as a dictionary
    
        """
    pass


'''
TODO:
1. Call run_optimization() function
2. Extract optimal parameters for each filter
3. Apply filters using optimal parameters
4. Display the optimized results in a 2x2 grid (noisy + 3 optimal filters)
5. Print the optimal parameters clearly
'''

'''
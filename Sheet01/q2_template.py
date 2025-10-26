import cv2
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
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
original_img_color = cv2.imread("bonn.jpg", cv2.IMREAD_COLOR)  # Load bonn.jpg
original_img_gray = cv2.imread("bonn.jpg", cv2.IMREAD_GRAYSCALE)   # Convert to grayscale
noisy_img = cv2.imread("bonn_noisy.jpg", cv2.IMREAD_COLOR) # Load bonn_noisy.jpg and convert to grayscale
noisy_img_float_01 = cv2.imread("bonn_noisy.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0  # Convert noisy image to float format (0-1)

# Calculate PSNR of noisy image
original_img_gray_float_01 = original_img_gray.astype(np.float32) / 255.0
psnr_noisy = cv2.PSNR(original_img_gray_float_01, noisy_img_float_01, R=1.0) # Using the built in function and R is the maximum pixel value 
print(f"PSNR of noisy image: {psnr_noisy} dB")

# Display original and noisy images
# TODO: Create a figure showing original and noisy images side by side
combined_images = cv2.hconcat([original_img_color, noisy_img])
cv2.imshow("Original vs Noisy ", combined_images)
cv2.waitKey(0)
cv2.destroyAllWindows()

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
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("Kernel size must be an odd integer")
    if sigma <= 0:
        raise ValueError("Sigma must be > 0")
    if image.dtype.kind not in ('f'):
        raise TypeError("Image must be in float [0,1] range")
    
    k = kernel_size // 2
    
    Y, X = np.mgrid[-k:k+1, -k:k+1] # Creating 2D coordinate grid
    
    # Gaussian formula
    kernel = np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))
    
    # Normalizing the kernel
    kernel_sum = np.sum(kernel, dtype=np.float64) 
    kernel = kernel / kernel_sum
    kernel = kernel.astype(np.float64)
    
    # Padding the image using reflect mode
    padded = np.pad(image, ((k, k), (k, k)), mode='reflect')
    
    # Applying convolution manually using nested loops
    output_img = np.empty_like(image, dtype=np.float32)
    H, W = image.shape
    for i in range(H):
        for j in range(W):
            patch = padded[i:i + kernel_size, j:j+ kernel_size]
            # Multiply the Gaussian kernel
            new_pixel = np.sum(patch * kernel, dtype=np.float64)
            output_img[i, j] = new_pixel
    
    np.clip(output_img, 0.0, 1.0, out=output_img) # Clipping values to [0,1] range
    return output_img

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
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("Kernel size must be an odd integer")
    if image.dtype.kind not in ('f'):
        raise TypeError("Image must be in float [0,1] range")
    
    k = kernel_size // 2
    
    padded = np.pad(image, ((k, k), (k, k)), mode='reflect') # Padding the image using the reflect mode
    
    # For each pixel extacting the window, calculatin the median and assiging the median value to the output pixel
    output_img = np.empty_like(image, dtype=np.float32)
    H, W = image.shape
    for i in range(H):
        for j in range(W):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            output_img[i, j] = np.median(window)
    
    np.clip(output_img, 0.0, 1.0, out=output_img)
    return output_img

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
    if d % 2 == 0 or d < 1:
        raise ValueError("d size must be an positive integer")
    if image.dtype.kind not in ('f',):
        raise TypeError("img must be float in [0,1].")
    if sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("Sigmas mut be > 0")
    
    radius = d // 2
    output_img = np.empty_like(image, dtype=np.float32)
    H, W = image.shape
    
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode='reflect')
    
    Y, X = np.mgrid[-radius:radius+1, -radius:radius+1] # Creating 2D coordinate grid
    g_space = np.exp(-(X**2 + Y**2) / (2.0 * sigma_space**2)).astype(np.float32)
    
    for i in range(H):
        for j in range(W):
            # Window
            window = padded[i:i+d, j:j+d]
            center = padded[i + radius, j + radius]
            
            # range weight : similar intesitites = higher weight
            diff = window - center
            g_range = np.exp((-diff**2) / (2.0 * sigma_color**2)).astype(np.float32)
            
            # combined bilateral weights
            weights_all = g_space * g_range
            
            weights_sum = np.sum(weights_all, dtype=np.float64)
            output_img[i, j] = np.sum(window * weights_all, dtype=np.float64) / weights_sum
    
    np.clip(output_img, 0.0, 1.0, out=output_img)
    return output_img


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

denoised_gaussian_cv2 = cv2.GaussianBlur(noisy_img_float_01, ksize=(K_DEFAULT, K_DEFAULT), sigmaX=S_DEFAULT, borderType=cv2.BORDER_REFLECT)
psnr_gaussian_cv2 = cv2.PSNR(original_img_gray_float_01, denoised_gaussian_cv2, R=1.0)

denoised_gaussian_custom = custom_gaussian_filter(noisy_img_float_01, kernel_size=K_DEFAULT, sigma=S_DEFAULT)
psnr_gaussian_custom = cv2.PSNR(original_img_gray_float_01, denoised_gaussian_custom, R=1.0)

# Display results here
print(f"PSNR (cv2.GaussianBlur) : {psnr_gaussian_cv2:.2f} dB")
print(f"PSNR (custom Gaussian) : {psnr_gaussian_custom:.2f} dB")

# Displaying the plots
plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.imshow(noisy_img_float_01, cmap='gray', vmin=0, vmax=1)
plt.title('Noisy')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(denoised_gaussian_cv2, cmap='gray', vmin=0, vmax=1)
plt.title(f'cv2 Gaussian\nPSNR: {psnr_gaussian_cv2:.2f} dB')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(denoised_gaussian_custom, cmap='gray', vmin=0, vmax=1)
plt.title(f'Custom Gaussian\nPSNR: {psnr_gaussian_custom:.2f} dB')
plt.axis('off')

plt.tight_layout()
plt.show()
# -------------------------- b) Median Filter --------------------------
print("b) Applying Median Filter...")
'''
TODO:
1. Apply Median filter using cv2.medianBlur()
2. Apply your custom Median filter
3. Calculate PSNR for both results
4. Display the results in a figure with 3 subplots
'''

denoised_median_cv2 = (cv2.medianBlur((noisy_img_float_01 * 255.0).astype(np.uint8), ksize=K_DEFAULT)).astype(np.float32) / 255.0
psnr_median_cv2 = cv2.PSNR(original_img_gray_float_01, denoised_median_cv2, R=1.0)

denoised_median_custom = custom_median_filter(noisy_img_float_01, kernel_size=K_DEFAULT)
psnr_median_custom = cv2.PSNR(original_img_gray_float_01, denoised_median_custom, R=1.0)


# Display results here
print(f"PSNR (cv2.medianBlur) : {psnr_median_cv2:.2f} dB")
print(f"PSNR (custom Median) : {psnr_median_custom:.2f} dB")

# Displaying the plots
plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.imshow(noisy_img_float_01, cmap='gray', vmin=0, vmax=1)
plt.title('Noisy')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(denoised_median_cv2, cmap='gray', vmin=0, vmax=1)
plt.title(f'cv2 Median\nPSNR: {psnr_median_cv2:.2f} dB')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(denoised_median_custom, cmap='gray', vmin=0, vmax=1)
plt.title(f'Custom Median\nPSNR: {psnr_median_custom:.2f} dB')
plt.axis('off')

plt.tight_layout()
plt.show()

# -------------------------- c) Bilateral Filter --------------------------
print("c) Applying Bilateral Filter...")
'''
TODO:
1. Apply Bilateral filter using cv2.bilateralFilter()
2. Apply your custom Bilateral filter (remember to scale sigma_color for 0-1 range)
3. Calculate PSNR for both results
4. Display the results in a figure with 3 subplots
'''

denoised_bilateral_cv2 = (cv2.bilateralFilter((noisy_img_float_01 * 255.0).astype(np.uint8), 
                                              d=D_DEFAULT, sigmaColor=SC_DEFAULT, 
                                              sigmaSpace=SS_DEFAULT, 
                                              borderType=cv2.BORDER_REFLECT)).astype(np.float32) / 255.0
psnr_bilateral_cv2 = cv2.PSNR(original_img_gray_float_01, denoised_bilateral_cv2, R=1.0)

denoised_bilateral_custom = custom_bilateral_filter(noisy_img_float_01, d=D_DEFAULT, sigma_color=(SC_DEFAULT / 255.0), sigma_space=SS_DEFAULT)
psnr_bilateral_custom = cv2.PSNR(original_img_gray_float_01, denoised_bilateral_custom, R=1.0)

# Display results here
print(f"PSNR (cv2.bilateralFilter): {psnr_bilateral_cv2:.2f} dB")
print(f"PSNR (custom bilateral):   {psnr_bilateral_custom:.2f} dB")


# Displaying the plots
plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.imshow(noisy_img_float_01, cmap='gray', vmin=0, vmax=1)
plt.title('Noisy')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(denoised_bilateral_cv2, cmap='gray', vmin=0, vmax=1)
plt.title(f'cv2 Bilateral\nPSNR: {psnr_bilateral_cv2:.2f} dB')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(denoised_bilateral_custom, cmap='gray', vmin=0, vmax=1)
plt.title(f'Custom Bilateral\nPSNR: {psnr_bilateral_custom:.2f} dB')
plt.axis('off')

plt.tight_layout()
plt.show()

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
# Computing PSNR for each of the filtered image using different filtering method
psnr_gaussian = psnr(original_img_gray_float_01, denoised_gaussian_custom, data_range=1.0)
psnr_median = psnr(original_img_gray_float_01, denoised_median_custom, data_range=1.0)
psnr_bilateral = psnr(original_img_gray_float_01, denoised_bilateral_custom, data_range=1.0)

psnr_scores = {
    "Gaussian" : psnr_gaussian,
    "Median" : psnr_median,
    "Bilateral" : psnr_bilateral
}
# Getting the best method and its score
best_method = max(psnr_scores, key=psnr_scores.get)
best_value = psnr_scores[best_method]

# Displaying the plots
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(noisy_img_float_01, cmap='gray', vmin=0, vmax=1)
plt.title('Noisy')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(denoised_gaussian_custom, cmap='gray', vmin=0, vmax=1)
plt.title(f'Gaussian\nPSNR: {psnr_gaussian:.2f} dB')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(denoised_median_custom, cmap='gray', vmin=0, vmax=1)
plt.title(f'Median\nPSNR: {psnr_median:.2f} dB')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(denoised_bilateral_custom, cmap='gray', vmin=0, vmax=1)
plt.title(f'Bilateral\nPSNR: {psnr_bilateral:.2f} dB')
plt.axis('off')

plt.tight_layout()
plt.show()

print("\nPSNR Results")
for name, value in psnr_scores.items():
    print(f"Method : {name}, Value : {value:.2f}")

print(f"\nBest performing filter : {best_method} ({best_value})")



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
    results = {
        "gaussian" : {"best_psnr" : -np.inf, "params" : None},
        "median" : {"best_psnr" : -np.inf, "params" : None},
        "bilateral" : {"best_psnr" : -np.inf, "params" : None},
    }

    # Gaussian params to test
    g_kernel_size = [3, 5, 7, 9, 11]
    g_sigams = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    # Median params to test
    m_kernel_sizes = [3, 5, 7, 9, 11]
    
    # Bilateral params to test
    b_d = [3, 5, 7, 9]
    b_sigma_colors = [60, 80, 100, 120, 150]
    b_sigma_space = [50, 60, 75, 90, 100]
    
    # Finding optimal params for Gaussian
    for k in g_kernel_size:
        for s in g_sigams:
            denoised = custom_gaussian_filter(noisy_img_float_01, kernel_size=k, sigma=s)
            score = psnr(original_img_gray_float_01, denoised, data_range=1.0)
            if score > results['gaussian']['best_psnr']:
                results['gaussian']['best_psnr'] = score
                results['gaussian']["params"] = {"kernel_size" : k, "sigma": s}
    
    # Finding optimal params for median
    for k in m_kernel_sizes:
        denoised = custom_median_filter(noisy_img_float_01, k)
        score = psnr(original_img_gray_float_01, denoised, data_range=1.0)
        if score > results['median']['best_psnr']:
                results['median']['best_psnr'] = score
                results['median']["params"] = {"kernel_size" : k}
    
    # Finding optimal params for bilateral
    for d in b_d:
        for sc in b_sigma_colors:
            for ss in b_sigma_space:
                denoised = custom_bilateral_filter(noisy_img_float_01, d=d, sigma_color=(sc / 255.0), sigma_space=ss)
                score = psnr(original_img_gray_float_01, denoised, data_range=1.0)
                if score > results['bilateral']['best_psnr']:
                    results['bilateral']['best_psnr'] = score
                    results['bilateral']["params"] = {"diameter" : d, "sigma_color": sc, "sigma_space":ss}
    return results

'''
TODO:
1. Call run_optimization() function
2. Extract optimal parameters for each filter
3. Apply filters using optimal parameters
4. Display the optimized results in a 2x2 grid (noisy + 3 optimal filters)
5. Print the optimal parameters clearly
'''
# optimal_results = run_optimization(original_img_gray_float_01, noisy_img_float_01)

# gaussian_params = optimal_results["gaussian"]["params"]
# median_params = optimal_results["median"]["params"]
# bilateral_params = optimal_results["bilateral"]["params"]
BEST_PARAMS_GAUSSIAN = {
    "kernel_size": 11,
    "sigma": 1.0
}

BEST_PARAMS_MEDIAN = {
    "kernel_size": 7
}

BEST_PARAMS_BILATERAL = {
    "diameter": 5,
    "sigma_color": 80,
    "sigma_space": 50
}
print("\nBest params")
print(f"Gaussian : {BEST_PARAMS_GAUSSIAN}")
print(f"Median : {BEST_PARAMS_MEDIAN}")
print(f"Bilateral : {BEST_PARAMS_BILATERAL}")

# Applying the filtering with optimal params
best_gauss_denoised = custom_gaussian_filter(noisy_img_float_01, kernel_size=BEST_PARAMS_GAUSSIAN["kernel_size"], sigma=BEST_PARAMS_GAUSSIAN["sigma"])
best_median_denoised = custom_median_filter(noisy_img_float_01, kernel_size=BEST_PARAMS_MEDIAN["kernel_size"])
best_bilateral_denoised = custom_bilateral_filter(noisy_img_float_01, 
                                                  d=BEST_PARAMS_BILATERAL["diameter"], 
                                                  sigma_color=(BEST_PARAMS_BILATERAL["sigma_color"] / 255.0), 
                                                  sigma_space=BEST_PARAMS_BILATERAL["sigma_space"])

# Displaying the result in plots
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(noisy_img_float_01, cmap='gray', vmin=0, vmax=1)
plt.title('Noisy')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(best_gauss_denoised, cmap='gray', vmin=0, vmax=1)
plt.title(f'Gaussian')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(best_median_denoised, cmap='gray', vmin=0, vmax=1)
plt.title(f'Median')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(best_bilateral_denoised, cmap='gray', vmin=0, vmax=1)
plt.title(f'Bilateral')
plt.axis('off')

plt.tight_layout()
plt.show()

# ==============================================================================
# 4. Discussion (Part f)
# ==============================================================================
print("""\
    The input image contained both Gaussian and Salt-and-Pepper noise and according to the PSNR values, the Gaussian filter achieved the highest score (26.28 dB),
    indicating the best overall noise reduction. This makes sense because the Gaussian filter smooths intensity variations and is very effective against Gaussian noise.
    The median filter peformed worse here because it can blur fine details when mixed noise is present. The bilateral filter gave a balanced result since it presevers edges
    while reducing noise.
    """)
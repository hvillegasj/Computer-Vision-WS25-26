import cv2
import numpy as np
import time

# ==============================================================================
# 0. Setup: Loading Image and Converting to Grayscale
# ==============================================================================
print("--- 0. Setup: Loading Image and Converting to Grayscale ---")

'''
TODO: Load the image 'bonn.jpg' and convert it to grayscale
'''

# Load image and convert to grayscale
original_img_color = cv2.imread("bonn.jpg", cv2.IMREAD_COLOR)  # Load 'bonn.jpg'
gray_img = cv2.imread("bonn.jpg", cv2.IMREAD_GRAYSCALE)            # Convert to grayscale

print(f"Image loaded successfully. Size: {gray_img.shape}")

# ==============================================================================
# 1. Calculate Integral Image (Part a)
# ==============================================================================
print("\n--- a) Calculating Integral Image ---")


def calculate_integral_image(img):
    """
    Calculate the integral image (summed area table).
    Each pixel contains the sum of all pixels above and to the left.
    
    Args:
        img: Input grayscale image
    
    Returns:
        Integral image with dimensions (height+1, width+1)
    
    TODO:
    1. Create an integral image array     
    2. Iterate through all pixels and compute integral values
    """
    if img.ndim != 2:
        raise ValueError("Image to be expected in grayscale")
    
    img_float_64 = img.astype(np.int64, copy=False)
    H, W = img_float_64.shape
    I = np.zeros((H + 1, W + 1), dtype=np.int64)
    
    for y in range(1, H + 1):
        row_sum = 0
        for x in range(1, W + 1):
            row_sum += img_float_64[y - 1, x - 1]
            I[y, x] = I[y - 1, x] + row_sum
    return I

# Calculate integral image
integral_img = calculate_integral_image(gray_img)  # Call calculate_integral_image()

print("Integral image calculated successfully.")
print(f"Integral image size: {integral_img.shape}")

# ==============================================================================
# 2. Compute Mean Using Integral Image (Part b)
# ==============================================================================
print("\n--- b) Computing Mean Using Integral Image ---")


def mean_using_integral(integral, top_left, bottom_right):
    """
    Calculate mean gray value using integral image.
    Time Complexity: O(1)

    Args:
        integral: The integral image
        top_left: (row, col) - top left corner of the region
        bottom_right: (row, col) - bottom right corner of the region
    
    Returns:
        Mean gray value of the region
    
    """
    # 1. Extract coordinates from top_left and bottom_right
    y1, x1 = top_left
    y2, x2 = bottom_right
    
    # 2. Adjust indices for integral image (remember it's 1-indexed)
    y1 += 1
    x1 += 1
    y2 += 1
    x2 += 1
    
    # 3. Return Sum / number_of_pixels
    region_sum = (
        integral[y2, x2]
        - integral[y1 - 1, x2]
        - integral[y2, x1 - 1]
        + integral[y1 - 1, x1 - 1]
    )
    
    num_pixels = (y2 - y1 + 1) * (x2 - x1 + 1)
    mean_value = region_sum / num_pixels
    
    return mean_value    

# Define region
top_left = (10, 10)
bottom_right = (60, 80)

# Calculate mean using integral image
mean_integral = mean_using_integral(integral_img, top_left, bottom_right)  # Call mean_using_integral()

print(f"Region: Top-left {top_left}, Bottom-right {bottom_right}")
print(f"Region size: {bottom_right[0] - top_left[0] + 1} x {bottom_right[1] - top_left[1] + 1} pixels")
print(f"Mean gray value (Integral Image Method): {mean_integral:.2f}")

# ==============================================================================
# 3. Compute Mean by Direct Summation (Part c)
# ==============================================================================
print("\n--- c) Computing Mean by Direct Summation ---")


def mean_by_direct_sum(img, top_left, bottom_right):
    """
    Calculate mean gray value by summing all pixels in region.
    Time Complexity: O(w * h) where w and h are region dimensions

    Args:
        img: The grayscale image
        top_left: (row, col) - top left corner of the region
        bottom_right: (row, col) - bottom right corner of the region
    
    Returns:
        Mean gray value of the region
    """

    #1. Extract the region from the image using array slicing
    y1, x1 = top_left
    y2, x2 = bottom_right

    #2. Calculate and return the mean of all pixels in the region
    region = img[y1:y2+1, x1:x2+1]
    
    mean_value = np.mean(region)
    
    return float(mean_value)


# Calculate mean using direct summation
mean_direct = mean_by_direct_sum(gray_img, top_left, bottom_right)  # Call mean_by_direct_sum()

print(f"Mean gray value (Direct Summation Method): {mean_direct:.2f}")

# ==============================================================================
# 4. Analyze Computational Complexity (Part d)
# ==============================================================================
print("\n--- d) Computational Complexity Analysis ---")

'''
TODO:
1. Benchmark both methods by running them multiple times (e.g., 100 iterations)
2. Measure execution time for both methods using time.perf_counter()
3. Compare the execution times
4. Verify that both methods produce the same result
5. Print the results:
   - Method name
   - Average execution time
   - Performance improvement factor


'''

# Benchmark parameters
iterations = 100

print(f"\nBenchmarking with {iterations} iterations...\n")

# TODO: Implement benchmarking code here
t0 = time.perf_counter()
for _ in range(iterations):
    v_integral = mean_using_integral(integral_img, top_left, bottom_right)
t1 = time.perf_counter()
time_integral = (t1 - t0) / iterations

t0 = time.perf_counter()
for _ in range(iterations):
    v_direct = mean_by_direct_sum(gray_img, top_left, bottom_right)
t1 = time.perf_counter()
time_direct = (t1 - t0) / iterations

# TODO: Display results
print(f"Direct sum : {time_direct*1e6:9.2f} microsecond / call")
print(f"Integral  : {time_integral*1e6:9.2f} microsecond / call")

print(f"Values (direct = {v_direct:.6f}, integral = {v_integral:.6f})")
 
# TODO: Print theoretical complexity explanation
print("""\
    Theoretical Complexity
    H,W be the image height/width; h,w the region height/width; 
    - Mean using Integral Image
        - Build integral image : O(H * W) (one time preprocessing)
        - Per region query: O(1)
        - Total for Q queries: O(H * W + Q)
        After cumulitative sum, any rectangle sum is constant time
    - Mean by Direct Summation
        - Per region query: O(h * w)
        - Total for Q queries: O(Q * h * w)
        Works grows lineraly with region area and number of queries
    """)
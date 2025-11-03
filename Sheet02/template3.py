# Template for Exercise 3 – Spatial and Frequency Domain Filtering
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(path : str):
    if  path == "":
        raise ValueError("There is no path to the image")
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def make_box_kernel(k):
    """
    Create a normalized k×k box filter kernel.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    
    kernel = np.ones((k,k), dtype=np.float64)
    kernel /= (k * k)
    return kernel


def make_gauss_kernel(k, sigma):
    """
    Create a normalized 2D Gaussian filter kernel of size k×k.
    """
    if k % 2 == 0 or k < 1:
        raise ValueError("Kernel size must be an odd integer")
    if sigma <= 0:
        raise ValueError("Sigma must be > 0")
    
    
    k_mod = k // 2
    
    Y, X = np.mgrid[-k_mod:k_mod+1, -k_mod:k_mod+1] # Creating 2D coordinate grid
    
    # Gaussian formula
    kernel = np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))
    
    # Normalizing the kernel
    kernel_sum = np.sum(kernel, dtype=np.float64) 
    kernel = kernel / kernel_sum
    kernel = kernel.astype(np.float64)
    
    return kernel


def conv2_same_zero(img, h):
    """
    Perform 2D spatial convolution using zero padding.
    Output should have the same size as the input image.
    (Do NOT use cv2.filter2D)
    """
    img = img.astype(np.float64)
    h = h.astype(np.float64)
    
    kernel_h, kernel_w = h.shape
    h_flip = h[::-1, ::-1]
    # Zero padding
    padded = np.pad(img, ((kernel_h // 2, kernel_h // 2), (kernel_w // 2, kernel_w // 2)), mode="constant", constant_values=0)
    
    output_img = np.empty_like(img, dtype=np.float64)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kernel_h, j:j+kernel_w]
            
            output_img[i,j] = np.sum(region * h, dtype=np.float64)
    
    return output_img

def freq_linear_conv(img, h):
    """
    Perform linear convolution in the frequency domain.
    (You can use numpy.fft)
    """
    img = img.astype(np.float64)
    h = h.astype(np.float64)
    
    H, W = img.shape
    kernel_h, kernel_w = h.shape
    
    #full linear conv size
    FH, FW = H + kernel_h - 1, W + kernel_w - 1
    """# Padding the image to full size
    img_padded = np.pad(img, ((0, kernel_h - 1), (0, kernel_w - 1)), mode="constant", constant_values=0)
    h_padded = np.pad(h, ((0, H - 1), (0, W - 1)), mode="constant", constant_values=0)
    
    # Shifting the kernel of the center is at (0,0)
    h_padded = np.fft.ifftshift(h_padded)
    
    # FFTS
    F_img = np.fft.fft2(img_padded, s=(FH, FW))
    F_h = np.fft.fft2(h_padded, s=(FH, FW))"""
    F_img = np.fft.fft2(img, s=(FH, FW))

    # --- Proper kernel embedding & centering ---
    # 1) put the small kernel at the CENTER of an FH×FW canvas
    h_big = np.zeros((FH, FW), dtype=np.float64)
    oy = (FH - kernel_h) // 2
    ox = (FW - kernel_w) // 2
    h_big[oy:oy + kernel_h, ox:ox + kernel_w] = h

    # 2) move the kernel center to (0,0) so FFT treats it as an impulse origin
    h_big = np.fft.ifftshift(h_big)

    # 3) FFT of the centered kernel
    F_h = np.fft.fft2(h_big)

    
    # Mulitply in frequency domain 
    F_out = F_img * F_h
    
    # Back to spaital
    y_full = np.fft.ifft2(F_out).real
    
    # Center crop back to SAME size
    start_y = (FH - H) // 2
    start_x = (FW - W) // 2
    y_same = y_full[start_y:start_y + H, start_x:start_x + W]
    
    return y_same

def compute_mad(a, b):
    """
    Compute Mean Absolute Difference (MAD) between two images.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    
    diff = np.abs(a - b)
    mad = np.mean(diff)
    
    return mad

def visualize(img, title, cmap="gray"):
    """
    Helper fuction to generate plots
    """
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()
# ==========================================================


if __name__ == "__main__":
    # 1. Load the grayscale image (e.g., lena.png)
    IMG_PATH = "data/lena.png"
    img = load_image(IMG_PATH)
    
    # 2. Construct 9×9 box and Gaussian kernels (same sigma)
    K = 9
    SIGMA = 2.0
    box_kernel = make_box_kernel(K)
    gauss_kernel = make_gauss_kernel(K, SIGMA)
    
    # 3. Apply both filters spatially (manual convolution)
    box_spatial = conv2_same_zero(img, h=box_kernel)
    gauss_spatial = conv2_same_zero(img, h=gauss_kernel)
    
    # 4. Apply both filters in the frequency domain
    box_freq = freq_linear_conv(img, h=box_kernel)
    gauss_freq = freq_linear_conv(img, h=gauss_kernel)
    
    # 5. Compute and print MAD between spatial and frequency outputs
    mad_box = compute_mad(box_spatial, box_freq)
    mad_gauss = compute_mad(gauss_spatial, gauss_freq)
    
    print(f"MAD (Box: spatial vs freq)   = {mad_box:.3e}")
    print(f"MAD (Gauss: spatial vs freq) = {mad_gauss:.3e}")
    
    # 6. Visualize all results (original, box/gaussian spatial, box/gaussian frequency, spectrum)
    
    visualize(img, "Original")
    visualize(box_spatial, "Box (Spatial)")
    visualize(box_freq, "Box (Frequency)")
    visualize(gauss_spatial, "Gauss (Spatial)")
    visualize(gauss_freq, "Gauss (Frequency)")
    
    # 7. Verify that MAD < 1×10⁻⁷ for both filters
    assert mad_box < 1e-7, f"MAD for box is too high : {mad_box}"
    assert mad_gauss < 1e-7, f"MAD for gaussian is too high : {mad_gauss}"
    print("MAD checks for both filters passed")
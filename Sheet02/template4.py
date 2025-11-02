# Template for Exercise 4 – NCC Stereo Matching

import cv2
import numpy as np
import matplotlib.pyplot as plt


WINDOW_SIZE = 11       # NCC patch size
MAX_DISPARITY = 64     # Maximum search range


def compute_manual_ncc_map(left_image, right_image, window_size, max_disparity):
    
    """Compute a dense disparity map using Normalized Cross-Correlation (NCC).
    
    Arguments:
        left_image, right_image : input grayscale stereo pair
        window_size             : size of the correlation window
        max_disparity           : maximum horizontal shift to consider

    Returns:
        disparity_map : computed disparity for each pixel (float32)"""
    
    if window_size % 2 == 0 or window_size < 1:
        raise ValueError("window_size must be a positive odd integer.")
    L = left_image.astype(np.float32, copy=False)
    R = right_image.astype(np.float32, copy=False)

    H, W = L.shape
    r = window_size // 2

    # pad both images so every center pixel has a full window
    # we are taking zero padding
    L_padding = np.pad(L, ((r, r), (r, r)), mode="constant", constant_values=0)
    R_padding = np.pad(R, ((r, r), (r, r)), mode="constant", constant_values=0)

    disp = np.zeros((H, W), dtype=np.float32)
    eps = 1e-8  # avoid /0

    # For each pixel, compare a patch in left to shifted patches in right
    # Convention: disparity d means R(y, x-d) matches L(y, x)
    for y in range(H):
        for x in range(W):
            # left patch (fixed)
            y0, y1 = y, y + 2*r + 1
            x0, x1 = x, x + 2*r + 1
            patchL = L_padding[y0:y1, x0:x1]
            meanL = patchL.mean()
            stdL  = patchL.std()

            best_ncc = -1.0
            best_d = 0

            # scan disparities
            # on the right image, the matching window is centered at (y, x-d)
            for d in range(max_disparity):
                xr0 = x - d
                xr1 = xr0 + 2*r + 1
                if xr0 < 0:
                    # if the right window would start < 0 even after padding,
                    # skip (no valid window for this disparity)
                    continue
                patchR = R_padding[y0:y1, xr0:xr1]

                meanR = patchR.mean()
                stdR  = patchR.std()

                denom = (stdL * stdR) + eps
                ncc = np.sum((patchL - meanL) * (patchR - meanR)) / denom

                if ncc > best_ncc:
                    best_ncc = ncc
                    best_d = d

            disp[y, x] = best_d

    return disp


def compute_mae(a, b, mask=None):
    """
    Compute Mean Absolute Error (MAE) between two disparity maps.
    Optionally, use a mask to exclude invalid pixels.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch")

    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)

    if mask is not None:
        m = mask.astype(bool)
        if m.sum() == 0:
            return np.nan
        return np.mean(np.abs(a[m] - b[m]))
    else:
        return np.mean(np.abs(a - b))


# ==========================================================


# TODO: 1. Load the stereo image pair (left.png, right.png) in grayscale
# TODO: 2. Call your NCC function to compute the manual disparity map
# TODO: 3. Compute a benchmark map using cv2.StereoBM_create with the same parameters
# TODO: 4. Visualize both maps and compare them qualitatively
# TODO: 5. Quantitatively compare both maps by computing MAE (Mean Absolute Error)
# TODO: 6. Ensure your manual implementation achieves MAE < 0.7 pixels

if __name__ == "__main__":
    # TODO 1: Load the stereo image pair (left.png, right.png) in grayscale
    left_path  = "data/left.jpg"   # change to your paths
    right_path = "data/right.jpg"

    left  = cv2.imread(left_path,  cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise FileNotFoundError("Could not load left/right images.")

    # TODO 2: Compute manual NCC disparity
    disp_manual = compute_manual_ncc_map(left, right, WINDOW_SIZE, MAX_DISPARITY)

    # TODO 3: Compute a benchmark map using cv2.StereoBM with the same parameters
    # StereoBM requires numDisparities to be a multiple of 16
    num_disp = int(np.ceil(MAX_DISPARITY / 16.0)) * 16

    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=WINDOW_SIZE)
    
    # NOTE: StereoBM expects 8-bit images
    disp_bm = stereo.compute(left, right).astype(np.float32) / 16.0  # fixed-point scale

    # TODO 4: Visualize both maps and compare qualitatively
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1); plt.imshow(left, cmap="gray"); plt.title("Left"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(disp_manual, cmap="inferno"); plt.title("Manual NCC"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(disp_bm, cmap="inferno"); plt.title("StereoBM"); plt.axis("off")
    plt.tight_layout(); plt.show()

    # TODO 5: Quantitatively compare using MAE
    # StereoBM often outputs negative disparities where invalid; mask them out
    valid_mask = disp_bm > 0
    mae = compute_mae(disp_manual, disp_bm, mask=valid_mask)
    print(f"MAE (Manual NCC vs StereoBM) on valid pixels: {mae:.3f} px")

    # TODO 6: Ensure MAE < 0.7 pixels
    assert mae < 0.7, f"MAE too high: {mae:.3f} px (target < 0.7)"

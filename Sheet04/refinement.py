import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt
import math
from segmentation import average_image, superpixel_segmentation_mask

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_edge_map(gray, sigma=1.0):
    """
    Compute gradient-based edge map and its squared magnitude.
    """
    # Slight smoothing is often helpful
    gray_blur = cv2.GaussianBlur(gray, (0, 0), sigma)

    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_sq = gx**2 + gy**2

    # Normalizing to [0,1] for stability
    grad_sq = grad_sq / (grad_sq.max() + 1e-6)

    return gx, gy, grad_sq


def internal_energy(v, alpha, beta):
    """
    Compute internal energy per vertex , so we can update locally.
    """
    # circular shifts
    v_ip1 = np.roll(v, -1, axis=0)
    v_im1 = np.roll(v,  1, axis=0)

    # first-order term (elasticity)
    d1 = v_ip1 - v                # shape (N,2)
    e1 = np.sum(d1**2, axis=1)    # ||v_{i+1}-v_i||^2

    # second-order term (curvature)
    d2 = v_ip1 - 2*v + v_im1
    e2 = np.sum(d2**2, axis=1)

    return alpha * e1 + beta * e2


def external_energy(v, grad_sq, lam):
    """
    External energy per vertex: -lambda * |∇I|^2 at that point.
    v : (N,2) array of (y,x) points
    grad_sq : gradient magnitude squared (normalized) of image
    """
    h, w = grad_sq.shape
    yy = np.clip(v[:, 0].round().astype(int), 0, h-1)
    xx = np.clip(v[:, 1].round().astype(int), 0, w-1)
    e_ext = -lam * grad_sq[yy, xx]
    return e_ext


def total_energy(v, grad_sq, alpha, beta, lam):
    """
    Total energy (scalar) for a contour v.
    """
    e_int = internal_energy(v, alpha, beta)
    e_ext = external_energy(v, grad_sq, lam)
    return np.sum(e_int + e_ext)


def greedy_step(v, grad_sq, alpha, beta, lam, search_radius=1):
    """
    One iteration of greedy snake update:
    for each vertex, try small displacements in a (2r+1)x(2r+1) window and
    keep the position that gives minimal local energy.
    """
    N = v.shape[0]
    v_new = v.copy()
    h, w = grad_sq.shape

    for i in range(N):
        best_pos = v[i].copy()
        best_energy = np.inf

        # neighbors indices for internal term
        i_prev = (i - 1) % N
        i_next = (i + 1) % N

        for dy in range(-search_radius, search_radius+1):
            for dx in range(-search_radius, search_radius+1):
                cand = v[i] + np.array([dy, dx])

                # stay inside image
                if (cand[0] < 0 or cand[0] >= h or
                    cand[1] < 0 or cand[1] >= w):
                    continue

                # temporary contour with moved point
                v_temp = v_new.copy()
                v_temp[i] = cand

                # compute local internal energy for i (using neighbors)
                local_v = v_temp[[i_prev, i, i_next], :]
                # reuse internal_energy on this 3-point
                e_int_local = internal_energy(local_v, alpha, beta)[1]

                # external energy at cand
                yy = int(round(cand[0]))
                xx = int(round(cand[1]))
                e_ext_local = -lam * grad_sq[yy, xx]

                e_total = e_int_local + e_ext_local

                if e_total < best_energy:
                    best_energy = e_total
                    best_pos = cand

        v_new[i] = best_pos

    return v_new

def initialize_from_mask(mask, n_points=80):
    """Take the largest contour of a binary mask and sample n_points along it
    """
    mask = (mask > 0).astype(np.uint8)
    
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    if len(contours) == 0:
        raise ValueError("No contour found in mask.")

    # Biggest contour
    cnt = max(contours, key=cv2.contourArea)
    cnt = cnt[:, 0, :]  # shape (M, 2) as (x, y)

    # Subsample to fixed number of snake points
    idx = np.linspace(0, len(cnt) - 1, n_points, dtype=int)
    pts = cnt[idx]   # (n_points, 2), (x, y)

    # Convert to (y, x) and float
    v0 = np.stack([pts[:, 1], pts[:, 0]], axis=1).astype(np.float32)
    return v0

def main():
    # 1) Load image
    img_path = "data/img_mosaic.tif"
    img_bgr = cv2.imread(img_path)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2) Initial segmentation (step 1, from segmentation.py)
    sp_mask = skimage.segmentation.slic(
        img, n_segments=400, compactness=18, start_label=0
    )
    average_color_mask = average_image(img, sp_mask)
    binary_mask = superpixel_segmentation_mask(img, sp_mask, average_color_mask, K=4)   # 0/1 mask of building candidates

    # 3) Pick one building candidate (largest connected component)
    binary_mask_uint8 = (binary_mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask_uint8, connectivity=8
    )

    # stats[i] = [x, y, w, h, area]
    if num_labels <= 1:
        raise ValueError("No building components found in initial mask.")

    areas = stats[1:, cv2.CC_STAT_AREA]  # skip label 0 (background)
    largest_label = 1 + np.argmax(areas)

    x, y, w, h, area = stats[largest_label]
    building_mask = (labels == largest_label).astype(np.uint8)

    # Crop ROI around this building (associated region)
    img_roi = img[y:y+h, x:x+w]
    mask_roi = building_mask[y:y+h, x:x+w]

    # 4) Compute edge map on grayscale ROI
    gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_RGB2GRAY)
    gx, gy, grad_sq = compute_edge_map(gray_roi, sigma=1.0)

    # 5) Initialize snake from building mask (inside ROI)
    v_roi = initialize_from_mask(mask_roi, n_points=100)

    # 6) Optimize snake 
    alpha = 0.1   # elasticity
    beta  = 0.5  # smoothness
    lam   = 2.0   # image force weight

    for it in range(80):
        v_roi = greedy_step(v_roi, grad_sq, alpha, beta, lam, search_radius=1)

    # 7) Map snake points back to full image coordinates
    v_global = v_roi.copy()
    v_global[:, 0] += y   # add row offset
    v_global[:, 1] += x   # add col offset

    # 8) Visualization
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(img)
    axs[0].set_title("Original image")
    axs[0].axis("off")

    axs[1].imshow(binary_mask, cmap="gray")
    axs[1].set_title("Initial building mask (step 1)")
    axs[1].axis("off")

    axs[2].imshow(img)
    axs[2].plot(v_global[:, 1], v_global[:, 0], "-r", linewidth=1)
    axs[2].set_title("Refined segmentation ")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

    

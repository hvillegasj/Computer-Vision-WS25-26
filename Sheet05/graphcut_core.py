import cv2
import numpy as np
import maxflow
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

ROOT = os.path.dirname(os.path.abspath(__file__))

def smooth_image(img, k_size=5, sigma=0):
    """
    Simple Gaussian smoothering to reduce noise
    """
    return cv2.GaussianBlur(img, (k_size, k_size), sigma)

def get_scribble_masks(label_img):
    """
    Extract foreground/background masks from the scribble image,
    using only NumPy. Assumes:
      - background is black (0,0,0)
      - there are two main scribble colors (FG and BG).
    """

    h, w = label_img.shape[:2]

    # All non-black pixels = some scribble
    b = label_img[:, :, 0]
    g = label_img[:, :, 1]
    r = label_img[:, :, 2]
    scribble_mask = (r != 0) | (g != 0) | (b != 0)

    if not np.any(scribble_mask):
        raise ValueError("No scribbles found in label image.")

    # Collect scribble colors
    scribble_pixels = label_img[scribble_mask].reshape(-1, 3)

    # Unique colors and their counts
    unique_colors, counts = np.unique(scribble_pixels, axis=0, return_counts=True)

    # Drop pure black if it slipped in
    non_black = ~np.all(unique_colors == 0, axis=1)
    unique_colors = unique_colors[non_black]
    counts = counts[non_black]

    if unique_colors.shape[0] < 2:
        raise ValueError("Less than two scribble colors found.")

    # Two most frequent scribble colors
    order = np.argsort(counts)[::-1]
    main_colors = unique_colors[order[:2]].astype(np.float32)  # (2,3)

    # Brighter color = FG (yellow-ish), darker = BG (red)
    intensities = main_colors.mean(axis=1)  # average B+G+R
    fg_idx = np.argmax(intensities)
    bg_idx = 1 - fg_idx

    fg_color = main_colors[fg_idx]
    bg_color = main_colors[bg_idx]

    # Classify each scribble pixel as closer to fg_color or bg_color
    fg_mask = np.zeros((h, w), dtype=bool)
    bg_mask = np.zeros((h, w), dtype=bool)

    pix = scribble_pixels.astype(np.float32)
    dist_fg = np.sum((pix - fg_color) ** 2, axis=1)
    dist_bg = np.sum((pix - bg_color) ** 2, axis=1)
    is_fg = dist_fg < dist_bg

    fg_mask[scribble_mask] = is_fg
    bg_mask[scribble_mask] = ~is_fg

    return fg_mask, bg_mask

def compute_iou(pred_mask, ground_truth_mask):
    """
    Compute Intersection over Union between two binary masks
    """
    pred = pred_mask > 0
    ground_truth = ground_truth_mask > 0
    
    intersection = np.logical_and(pred, ground_truth).sum()
    union = np.logical_or(pred, ground_truth).sum()
    
    return intersection / union

# Main Graph Cut Algorithm
class GraphCut:
    def __init__(self, img, labels, n_components=5, lambda_smooth=50.0):
        self.original_img = img
        self.img = smooth_image(img)
        self.labels = labels
        self.n_components = n_components
        self.lambda_smooth = lambda_smooth
        
        self.height, self.width = img.shape[:2]
        
        # Scribble masks
        self.fg_mask, self.bg_mask = get_scribble_masks(self.labels)
        
        # GMMs for foreground and background
        self.fg_gmm = None
        self.bg_gmm = None
    
    def fit_gmms(self):
        """
        Fit separate GMMs for foreground and background from scibbels
        """

        # Collect pixels
        fg_pixels = self.img[self.fg_mask].reshape(-1, 3)
        bg_pixels = self.img[self.bg_mask].reshape(-1, 3)
        
        self.fg_gmm = GaussianMixture(n_components=self.n_components, covariance_type="full", random_state=0).fit(fg_pixels)
        self.bg_gmm = GaussianMixture(n_components=self.n_components, covariance_type="full", random_state=0).fit(bg_pixels)

    def compute_unary_cost(self):
        """Compute unary terms for each pixel for FG and BG
        """
        flat_img = self.img.reshape(-1, 3).astype(np.float32)
        
        # Computing for every pixel log p(x | FG) and log p( x | BG)
        # D_fg(x) = unary cost of assigning pixel x to foreground  
        # D_bg(x) = unary cost of assigning pixel x to background

        D_fg = -(self.fg_gmm.score_samples(flat_img))
        D_bg = -(self.bg_gmm.score_samples(flat_img))
        
        D_fg = D_fg.reshape(self.height, self.width)
        D_bg = D_bg.reshape(self.height, self.width)
        
        # Forgground scribbels : cannot be background
        D_fg[self.fg_mask] = 0.0
        D_bg[self.fg_mask] = np.inf
        
        # Background scribbels : cannot be foreground
        D_fg[self.bg_mask] = np.inf
        D_bg[self.bg_mask] = 0.0
        
        return D_fg, D_bg
    
    def compute_beta(self):
        """
        Computing beta which controls the pairwise weight. It decreses when the color differences between neighboring pixels increases
        """
        img = self.img.astype(np.float32)
        
        differences = []
        
        # Horizontal neighbors
        diff_h = img[:, 1:, :] - img[:, :-1, :]
        differences.append(np.sum(diff_h ** 2, axis=2))
        
        # Vertical negihbors
        diff_v = img[1:, :, :] - img[:-1, :, :]
        differences.append(np.sum(diff_v ** 2, axis=2))
        
        avg_diff = np.mean(np.concatenate([d.flatten() for d in differences]))
        
        beta = 1.0 / (2.0 * avg_diff + 1e-6)
        
        return beta
    
    def build_graph_and_segment(self):
        """
        Build the graph  with unary and pariwise terms, run maxflow and return a binary segmentation mask
        """
        # Fit GMMs
        self.fit_gmms()
        
        # Computing unary terms
        D_fg, D_bg = self.compute_unary_cost()
        
        # Pairwise terms
        beta = self.compute_beta()
        gamma = self.lambda_smooth
        
        g = maxflow.Graph[float]()
        node_ids = g.add_grid_nodes((self.height, self.width))
        
        # Adding the edge weights
        g.add_grid_tedges(node_ids, D_fg, D_bg)
        
        img = self.img.astype(np.float32)
        
        # Horizontal edges
        for y in range(self.height):
            for x in range(self.width - 1):
                color_diff = img[y, x] - img[y , x + 1]
                dist_squared = np.dot(color_diff, color_diff)
                w = gamma * np.exp(-beta * dist_squared)
                
                g.add_edge(node_ids[y , x], node_ids[y, x + 1], w, w)
        
        # Vertical edges
        for y in range(self.height - 1):
            for x in range(self.width):
                color_diff = img[y, x] - img[y + 1, x]
                dist_squared = np.dot(color_diff, color_diff)
                w = gamma * np.exp(-beta * dist_squared)
                
                g.add_edge(node_ids[y, x], node_ids[y + 1, x], w, w)
        
        # Run max-flow / min-cut
        g.maxflow()
        
        # Get segmentation
        segments = g.get_grid_segments(node_ids)
        seg_mask = np.uint8(segments) * 255 # FG = 255 and BG = 0
        
        return seg_mask

def main():
    #Import the images
    image_names = [
    "106024",
    "208001",
    "aero_2008_002358",
    "bike_2007_005878",
    "person7",
    "scissors"
    ]
    
    iou_scores = []
    for name in image_names:
        img_path = os.path.join(ROOT, "dataset", "images", f"{name}.jpg")
        labels_path = os.path.join(ROOT, "dataset", "images-labels", f"{name}-anno.png")
        ground_truth_path = os.path.join(ROOT, "dataset", "images-gt", f"{name}.png")
        
        if not os.path.exists(img_path):
            print(f"Error: {img_path} not found!")

        if not os.path.exists(labels_path):
            print(f"Error: {labels_path} not found!")

        if not os.path.exists(ground_truth_path):
            print(f"Error: {ground_truth_path} not found!")
            
        img = cv2.imread(img_path)
        labels = cv2.imread(labels_path)
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        
        segmenter = GraphCut(img, labels, n_components=5, lambda_smooth=80.0)
        
        seg_mask = segmenter.build_graph_and_segment()
        
        # Compute IoU
        iou_score = compute_iou(seg_mask, ground_truth)
        print(f"IoU score : {iou_score:.3f}")
        iou_scores.append(iou_score)
    
    avg_iou = sum(iou_scores) / len(iou_scores)
    print("Average IoU:", avg_iou)
    
    # Show results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Prediction")
    plt.imshow(seg_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Ground Truth")
    plt.imshow(ground_truth, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    
    
if __name__ == "__main__":
    main()
import argparse
import cv2
import numpy as np
import maxflow
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import tkinter as tk
from tkinter import filedialog

from datetime import datetime

# you can use GUI-related libraries if needed


ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------- Utility functions ---------------------------- #

def smooth_image(img, k_size=5, sigma=0):
    """
    Simple Gaussian smoothing to reduce noise.
    """
    return cv2.GaussianBlur(img, (k_size, k_size), sigma)



def compute_iou(pred_mask, ground_truth_mask):
    """
    Compute Intersection over Union between two binary masks.
    pred_mask, ground_truth_mask: 0/255 or 0/1
    """
    pred = pred_mask > 0
    ground_truth = ground_truth_mask > 0

    intersection = np.logical_and(pred, ground_truth).sum()
    union = np.logical_or(pred, ground_truth).sum()

    if union == 0:
        return 0.0
    return intersection / union


# ---------------------------- GraphCut core ---------------------------- #

class GraphCut:
    def __init__(self, img, fg_mask, bg_mask,  n_components=5, lambda_smooth=50.0):
        self.original_img = img
        self.img = smooth_image(img)
        
        self.n_components = n_components
        self.lambda_smooth = lambda_smooth

        self.height, self.width = img.shape[:2]

        # User-provided hard constraints
        self.fg_mask = fg_mask.astype(bool)
        self.bg_mask = bg_mask.astype(bool)

        if not self.fg_mask.any() or not self.bg_mask.any():
            raise ValueError("Need at least one FG and one BG scribble.")
        # GMMs for foreground and background
        self.fg_gmm = None
        self.bg_gmm = None

    def fit_gmms(self):
        """
        Fit separate GMMs for foreground and background from scribbles.
        """
        # Collect pixels
        fg_pixels = self.img[self.fg_mask].reshape(-1, 3)
        bg_pixels = self.img[self.bg_mask].reshape(-1, 3)

        k_fg = min(self.n_components, len(fg_pixels))
        k_bg = min(self.n_components, len(bg_pixels))

        if k_fg < 1 or k_bg < 1:
            raise ValueError("Not enough FG/BG pixels to fit GMMs.")
        
        self.fg_gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            random_state=0,
        ).fit(fg_pixels)

        self.bg_gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            random_state=0,
        ).fit(bg_pixels)

    def compute_unary_cost(self):
        """
        Compute unary terms for each pixel for FG and BG.
        """
        flat_img = self.img.reshape(-1, 3).astype(np.float32)

        # D_fg(x) = unary cost of assigning pixel x to foreground
        # D_bg(x) = unary cost of assigning pixel x to background
        D_fg = -(self.fg_gmm.score_samples(flat_img))
        D_bg = -(self.bg_gmm.score_samples(flat_img))

        D_fg = D_fg.reshape(self.height, self.width)
        D_bg = D_bg.reshape(self.height, self.width)

        # Foreground scribbles: cannot be background
        D_fg[self.fg_mask] = 0.0
        D_bg[self.fg_mask] = np.inf

        # Background scribbles: cannot be foreground
        D_fg[self.bg_mask] = np.inf
        D_bg[self.bg_mask] = 0.0   # <-- fixed line

        return D_fg, D_bg

    def compute_beta(self):
        """
        Compute beta which controls the pairwise weight. It decreases when
        the color differences between neighboring pixels increase.
        """
        img = self.img.astype(np.float32)

        differences = []

        # Horizontal neighbors
        diff_h = img[:, 1:, :] - img[:, :-1, :]
        differences.append(np.sum(diff_h ** 2, axis=2))

        # Vertical neighbors
        diff_v = img[1:, :, :] - img[:-1, :, :]
        differences.append(np.sum(diff_v ** 2, axis=2))

        avg_diff = np.mean(np.concatenate([d.flatten() for d in differences]))
        beta = 1.0 / (2.0 * avg_diff + 1e-6)

        return beta

    def build_graph_and_segment(self):
        """
        Build the graph with unary and pairwise terms, run maxflow
        and return a binary segmentation mask (0/255).
        """
        # Fit GMMs
        self.fit_gmms()

        # Compute unary terms
        D_fg, D_bg = self.compute_unary_cost()

        # Pairwise terms
        beta = self.compute_beta()
        gamma = self.lambda_smooth

        g = maxflow.Graph[float]()
        node_ids = g.add_grid_nodes((self.height, self.width))

        # Add terminal edges (unary)
        g.add_grid_tedges(node_ids, D_bg, D_fg)  # src=BG, sink=FG OR vice-versa, consistent

        img = self.img.astype(np.float32)

        # Horizontal edges
        for y in range(self.height):
            for x in range(self.width - 1):
                color_diff = img[y, x] - img[y, x + 1]
                dist_squared = float(np.dot(color_diff, color_diff))
                w = gamma * np.exp(-beta * dist_squared)
                g.add_edge(node_ids[y, x], node_ids[y, x + 1], w, w)

        # Vertical edges
        for y in range(self.height - 1):
            for x in range(self.width):
                color_diff = img[y, x] - img[y + 1, x]
                dist_squared = float(np.dot(color_diff, color_diff))
                w = gamma * np.exp(-beta * dist_squared)
                g.add_edge(node_ids[y, x], node_ids[y + 1, x], w, w)

        # Run max-flow / min-cut
        g.maxflow()

        # Get segmentation: True = sink/FG (depending on tedge convention)
        segments = g.get_grid_segments(node_ids)
       #seg_mask = np.uint8(segments) * 255  # FG = 255, BG = 0
        seg_mask = (~segments).astype(np.uint8) * 255
        return seg_mask


# ======================= Interactive App ==================== #

class InteractiveGraphCutApp:
    def __init__(self, image_path, gt_path=None, brush_radius=4, lambda_smooth=80.0):
        # Load image
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        if image_path is None:
            # Start with a blank canvas; user presses 'o' to pick an image.
            self.img = np.zeros((480, 640, 3), dtype=np.uint8)
            self.h, self.w = self.img.shape[:2]
            self.gt_mask = None
        else:
            self.img = cv2.imread(image_path)
            if self.img is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
            self.h, self.w = self.img.shape[:2]

        # "Labels" image, initially all black (no scribbles)
        self.fg_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self.bg_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        self.scribble_vis = np.zeros_like(self.img, dtype=np.uint8)
        # For visualization
        self.display = self.img.copy()

        # Current segmentation mask (0/255), None until we run GraphCut
        self.seg_mask = None

        # Interaction state
        self.drawing = False
        self.mode = "FG"  # "FG", "BG", "ERASE"
        self.brush_radius = brush_radius
        self.lambda_smooth = lambda_smooth
        self.window_name = "Interactive GraphCut"

        # Colors for scribbles (BGR)
        self.fg_color = (0, 255, 255)   # yellow
        self.bg_color = (0, 0, 255)     # red
        self.erase_color = (0, 0, 0)    # black

        # Ground-truth (for IoU)
        self.gt_path = gt_path
        self.gt_mask = self._load_gt_mask(gt_path) if gt_path is not None else None

        self._print_help()

    def _load_gt_mask(self, path):
        gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f"[WARN] Could not read GT mask at {path}, IoU will be skipped.")
            return None
        gt_bin = (gt > 0).astype(np.uint8) * 255
        return gt_bin

    @staticmethod
    def _print_help():
        print(
            """
[Interactive GraphCut]

Mouse:
  - Left button + drag: draw scribbles

Keys:
  f      : Foreground scribble (yellow)
  b      : Background scribble (red)
  e      : Eraser (remove scribbles, set back to black)
  g      : Run / update GraphCut with current scribbles
  r      : Reset scribbles + segmentation
  s      : Save current binary mask (PNG).
           If GT mask is provided, IoU is printed.
  o      : Open or Upload the image.
  t      : Upload the grouth truth image
  k      : Brush thickness 
  q or ESC: Quit
"""
        )

    # ---------------- Mouse handling ---------------- #
    def pick_image_file(self, title="Select image"):
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        root.destroy()
        return path
    
    def load_new_image(self, image_path):
        """Loading the original image
        """
        
        img = cv2.imread(image_path)
        if img is None:
            print("[WARN] Could not read image:", image_path)
            return

        self.image_path = image_path
        self.img = img
        self.h, self.w = img.shape[:2]

        # re-init masks for the new size
        self.fg_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self.bg_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self.scribble_vis = np.zeros_like(self.img, dtype=np.uint8)

        self.seg_mask = None
        self.display = self.img.copy()
        
        self.gt_mask = None
        self.gt_path = None

        self.update_display()
        print("[INFO] Loaded image:", image_path)
     
    def load_gt_from_path(self, gt_path):
        """Loading the ground truth image
        """
        
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print("[WARN] Could not read GT:", gt_path)
            self.gt_mask = None
            return
        self.gt_mask = (gt > 0).astype(np.uint8) * 255
        print("[INFO] Loaded GT:", gt_path)

        # If segmentation already exists, print IoU immediately
        if self.seg_mask is not None:
            iou = compute_iou(self.seg_mask, self.gt_mask)
            print(f"[INFO] IoU with GT: {iou:.4f}")

    def open_brush_slider(self):
        """GUI dropdown slider
        """
        win = tk.Tk()
        win.title("Brush Size")

        def update(val):
            self.brush_radius = int(val)
            print(f"[INFO] Brush radius: {self.brush_radius}")

        slider = tk.Scale(win, from_=1, to=50,
                        orient=tk.HORIZONTAL,
                        label="Brush Radius",
                        command=update)
        slider.set(self.brush_radius)
        slider.pack()
        win.mainloop()

    def set_mode(self, mode):
        self.mode = mode
        print(f"[INFO] Mode set to: {self.mode}")

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self._draw_at(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self._draw_at(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self._draw_at(x, y)

    def _draw_at(self, x, y):
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return

        if self.mode == "FG":
            cv2.circle(self.scribble_vis, (x, y), self.brush_radius, self.fg_color, -1)
            cv2.circle(self.fg_mask, (x, y), self.brush_radius, 1, -1)
            cv2.circle(self.bg_mask, (x, y), self.brush_radius, 0, -1)

        elif self.mode == "BG":
            cv2.circle(self.scribble_vis, (x, y), self.brush_radius, self.bg_color, -1)
            cv2.circle(self.bg_mask, (x, y), self.brush_radius, 1, -1)
            cv2.circle(self.fg_mask, (x, y), self.brush_radius, 0, -1)

        elif self.mode == "ERASE":
            cv2.circle(self.scribble_vis, (x, y), self.brush_radius, (0, 0, 0), -1)
            cv2.circle(self.fg_mask, (x, y), self.brush_radius, 0, -1)
            cv2.circle(self.bg_mask, (x, y), self.brush_radius, 0, -1)

        self.update_display()



    # ====================== GraphCut + visualization ========================= #

    def run_graphcut(self):
        # Check if we have any scribbles
        try:
            segmenter = GraphCut(
                img=self.img,
                fg_mask=self.fg_mask,
                bg_mask=self.bg_mask,
                n_components=5,
                lambda_smooth=self.lambda_smooth,
            )
            seg_mask = segmenter.build_graph_and_segment()
            self.seg_mask = seg_mask
            self.update_display()
            print("[INFO] GraphCut done.")
            
        except ValueError as e:
            print(f"[ERROR] {e}")

    def update_display(self):
        # Start from original image
        vis = self.img.copy()

        # If segmentation exists, overlay it
        if self.seg_mask is not None:
            fg = self.seg_mask > 0
            overlay = vis.copy()
            overlay[fg] = (0, 255, 0)  # green for FG
            alpha = 0.4
            vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

        # Overlay scribbles
        scribble_mask = np.any(self.scribble_vis != 0, axis=2)
        vis[scribble_mask] = self.scribble_vis[scribble_mask]

        self.display = vis

    # ---------------- Reset & Save ---------------- #

    def reset(self):
        print("[INFO] Resetting scribbles and segmentation.")
        self.fg_mask[:] = False
        self.bg_mask[:] = False
        self.scribble_vis[:] = 0
        self.seg_mask = None
        self.update_display()

    def save_mask(self):
        if self.seg_mask is None:
            print("[WARN] No segmentation yet. Run GraphCut with 'g' first.")
            return

        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(ROOT, "interactive_masks")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base_name}_interactive_{timestamp}.png")

        cv2.imwrite(out_path, self.seg_mask)
        print(f"[INFO] Saved mask to: {out_path}")

        if self.gt_mask is not None:
            iou = compute_iou(self.seg_mask, self.gt_mask)
            print(f"[INFO] IoU with GT: {iou:.4f}")

        cv2.imshow(self.window_name, self.display)
        cv2.waitKey(1)
    # ---------------- Main loop ---------------- #

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        self.update_display()

        while True:
            cv2.imshow(self.window_name, self.display)
            key = cv2.waitKey(20) & 0xFF

            if key == 27 or key == ord("q"):
                print("[INFO] Exiting.")
                break
            elif key == ord("f"):
                self.set_mode("FG")
            elif key == ord("b"):
                self.set_mode("BG")
            elif key == ord("e"):
                self.set_mode("ERASE")
            elif key == ord("g"):
                self.run_graphcut()
            elif key == ord("r"):
                self.reset()
            elif key == ord("s"):
                self.save_mask()
            elif key == ord("o"):
                path = self.pick_image_file("Upload / Open Image")
                if path:
                    self.load_new_image(path)
            elif key == ord("t"):
                path = self.pick_image_file("Upload / Open Ground Truth Mask")
                if path:
                    self.load_gt_from_path(path)
            elif key == ord("i"):
                if self.seg_mask is None or self.gt_mask is None:
                    print("[WARN] Need both segmentation and GT loaded.")
                elif self.seg_mask.shape != self.gt_mask.shape:
                    print("[WARN] Shape mismatch. Skipping IoU.")
                else:
                    print(f"[INFO] IoU with GT: {compute_iou(self.seg_mask, self.gt_mask):.4f}")
            elif key == ord('k'):
                self.open_brush_slider()



        cv2.destroyAllWindows()


# ---------------------------- CLI entry point ---------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive GraphCut using your GMM+maxflow core")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to input image.")
    parser.add_argument("--gt", type=str, default=None, help="Path to ground-truth binary mask (optional, for IoU).")
    parser.add_argument("--brush_radius", type=int, default=4, help="Radius of scribble brush in pixels.")
    parser.add_argument("--lambda_smooth", type=float, default=80.0, help="Smoothness weight (gamma).")
    return parser.parse_args()


def main():
    args = parse_args()
    app = InteractiveGraphCutApp(
        image_path=args.image,
        gt_path=args.gt,
        brush_radius=args.brush_radius,
        lambda_smooth=args.lambda_smooth,
    )
    app.run()


if __name__ == "__main__":
    main()

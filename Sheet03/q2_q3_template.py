"""
Task 2: Hough Transform for Circle Detection
Task 3: Mean Shift for Peak Detection in Hough Accumulator
Template for MA-INF 2201 Computer Vision WS25/26
Exercise 03
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os


def myHoughCircles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_steps):
    """
    Your implementation of HoughCircles
    
    Args:
        edges: single-channel binary source image (e.g: edges)
        min_radius: minimum circle radius
        max_radius: maximum circle radius
        param threshold: minimum number of votes to consider a detection
        min_dist: minimum distance between two centers of the detected circles. 
        r_ssz: stepsize of r
        theta_steps: steps to consider of theta
        return: list of detected circles as (a, b, r, v), accumulator as [r, y_c, x_c]
    """
    max_radius = min(max_radius, int(np.linalg.norm(edges.shape)))

    edges_points = np.transpose(np.array(np.nonzero(edges)))
    h, w = edges.shape

    #Create the array of radius and angles to be considered
    usable_radius = np.linspace(min_radius, max_radius, np.floor((max_radius - min_radius)/r_ssz).astype(int))
    number_radius = usable_radius.shape[0]

    usable_angles=np.linspace(0, 2*np.pi, theta_steps, endpoint=False)

    #Create the matrix to store the votes
    accumulator = np.zeros((h, w, number_radius) ,dtype=np.float32)

    for point in edges_points:
        for r_idx, r in enumerate(usable_radius):
            a_coords = np.round(point[0] - r * np.sin(usable_angles)).astype(int)
            b_coords = np.round(point[1] - r * np.cos(usable_angles)).astype(int)

            #Check that the coordinate is in bounds
            valid_mask = (a_coords >= 0) & (a_coords < h) & \
                         (b_coords >= 0) & (b_coords < w)

            # Increment votes only for valid coordinates
            for a, b in zip(a_coords[valid_mask], b_coords[valid_mask]):
                accumulator[a, b, r_idx] += 1.0

    #Whe threshold the accumulator to focus on strong candidates
    vertical_coord, horizontal_coord, potential_r = np.where(accumulator>=threshold)

    #We build a list that element wise stores the centre, radius and votes that have the elements
    #of our treshholded accumulator
    votes = accumulator[vertical_coord, horizontal_coord, potential_r]

    list_of_centres=np.transpose(np.stack([vertical_coord,horizontal_coord,potential_r,votes]))
    
    #We order the centres by votes, so we get the strongest centres in the top of the list
    list_of_centres=list_of_centres[np.argsort(list_of_centres[:, 3])[::-1]]

    detected_circles=[]

    for y_coord, x_coord, rad_index, vote_count in list_of_centres:
        #Until we check distance to previously accepted circles we assume is a new circle
        consider_circle=True 

        #Check distance to previous circles
        for circle in detected_circles:
            if ((y_coord - circle[0])**2 - (x_coord - circle[1])**2)<min_dist**2:
                consider_circle=False
                break
        
        if consider_circle:
            detected_circles.append([y_coord, x_coord, usable_radius[rad_index.astype(int)], vote_count])



    return  detected_circles, accumulator


def drawCirclesOnImage(img, peaks, radii_bins=None, color=(0, 255, 0), thickness=2):
    """Draws circles on existed grayscale/BGR image
    """

    # Converting grayscale to BGR so colors can be drawn
    if len(img.shape) == 2:
        vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = img.copy()
    
    for c in peaks:
        # Center coordinates
        x = int(c[0])
        y = int(c[1])
        
        if len(c) >= 3:
            r_idx = int(c[2]) # radius index
        else:
            continue
        # If radii_bins is provided, map r_idx -> true radius value
        if radii_bins is not None:
            r = int(radii_bins[r_idx])
        else:
            r = r_idx   # fallback if bins not provided
        
        # Draw main circle
        cv2.circle(vis_img, (x, y), r, color, thickness)
        # Draw a small dot at the center
        cv2.circle(vis_img, (x, y), 2, (0,0,255), -1)
    
    return vis_img

def myMeanShift(accumulator, bandwidth, threshold=None):
    """
    Find peaks in Hough accumulator using mean shift.
    
    Args:
        accumulator: 3D Hough accumulator (n_radii, h, w)
        bandwidth: Bandwidth for mean shift
        threshold: Minimum value to consider (if None, use fraction of max)
        
    Returns:
        peaks: List of (x, y, r_idx, value) tuples
    """
    
    accumulator = np.transpose(accumulator, (2, 0, 1))
    
    n_r, h, w = accumulator.shape
    
    # Computing threshold if None is passed as an argument
    if threshold is None:
        m = accumulator.max()
        if m <= 0:
            return []
        threshold = 0.5 * m
    
    # Collecting seed points
    seeds = np.argwhere(accumulator >= threshold)
    MAX_SEEDS = 5000 # safey cap on the number of seeds because it over 2.2 million seeds
    
    # Keeping only the strongest seeds
    if len(seeds) > MAX_SEEDS:
        # Get accumulator values for all seed positions
        vals = np.array([accumulator[int(r), int(y), int(x)] for (r, y, x) in seeds], dtype=float)
        # Keep indices of top MAX_SEEDS values
        keep = np.argpartition(vals, -MAX_SEEDS)[-MAX_SEEDS:]
        
        seeds = [seeds[i] for i in keep]
    print("Total number of seeds:", len(seeds))
    if not seeds:
        return []
    
    peaks = []
    
    max_iter = 30 # Upper bound on mean shift steps
    stop_eps = 1e-2 # Stopping condition when movement is tiny
    
    # Mean shifting from each seed
    for r, y, x in seeds:
        r, y, x = float(r), float(y), float(x)
        
        for _ in range(max_iter):
            # windows
            r_min = max(0, int(r - bandwidth))
            r_max = min(n_r - 1, int(r + bandwidth))
            
            y_min = max(0, int(y - bandwidth))
            y_max = min(h - 1, int(y + bandwidth))
            
            x_min = max(0, int(x - bandwidth))
            x_max = min(w - 1, int(x + bandwidth))
            
            sub_accumulator = accumulator[r_min:r_max+1, y_min:y_max+1, x_min:x_max+1]
            
            # Creating 3D coordinate grids
            r_axis = np.arange(r_min, r_max+1)[:, None, None]
            y_axis = np.arange(y_min, y_max+1)[None, :, None]
            x_axis = np.arange(x_min, x_max+1)[None, None, :]
            
            #Squared distances
            dist2 = (r_axis - r)**2 + (y_axis - y)**2 + (x_axis - x)**2
            Wg = np.exp(-0.5 * (dist2 / max(float(bandwidth)**2 , 1e-12)))
            
            weights = sub_accumulator * Wg
            S = weights.sum()
            
            if S == 0:
                break
            
            r_new = (weights * r_axis).sum() / S
            y_new = (weights * y_axis).sum() / S
            x_new = (weights * x_axis).sum() / S
            
            shift = np.sqrt((r_new - r)**2 + (y_new - y)**2 + (x_new - x)**2)
            
            r, y , x = r_new, y_new, x_new
            
            #Convergence
            if shift < stop_eps:
                break
        
        r, y, x = int(round(r)), int(round(y)), int(round(x))
        
        # Clip to array bounds
        r = np.clip(r, 0, n_r - 1)
        y = np.clip(y, 0, h - 1)
        x = np.clip(x, 0, w - 1)
        
        val = accumulator[r, y, x]
        peaks.append((x, y, r, val))
    
    return peaks

def main():
    
    print("=" * 70)
    print("Task 2: Hough Transform for Circle Detection")
    print("=" * 70)
        
    img_path = 'data/coins.jpg'
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        return
        
    # Load image and convert to grayscale
    img_original=cv2.imread(img_path)
    img=cv2.cvtColor(img_original,cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    lower_threshold, higher_threshold = 0.33,0.65
    img_edges=cv2.Canny(img, lower_threshold*255, higher_threshold*255)
    
    # Detect circles - parameters tuned for coins image
    print("\nDetecting circles...")
    min_radius = 10
    max_radius = 60
    threshold = 9
    min_dist = 60
    r_ssz = 2
    theta_ssz = 310
    
    detect_circles , accumulator = myHoughCircles(img_edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz)

    circles_for_draw = [(int(x), int(y), float(r), float(v)) for (y, x, r, v) in detect_circles] # Just converting it in the required format for visulaization
    vis_canvas = drawCirclesOnImage(img_original, circles_for_draw, radii_bins=None)

    # Visualize detected circles
    fig, axes = plt.subplots(2, 3, figsize=(12, 4)) 

    axes[0][0].imshow(img_original)
    axes[0][0].set_title('Original image')

    axes[0][1].imshow(img_edges, cmap='gray')
    axes[0][1].set_title('Edge map')

    axes[0][2].imshow(vis_canvas, cmap='gray')
    axes[0][2].set_title('Circles detected')

    axes[1][0].imshow(accumulator[:,:,10], cmap='gray')
    axes[1][0].set_title('Accumulator r=20')

    axes[1][1].imshow(accumulator[:,:,20], cmap='gray')
    axes[1][1].set_title('Accumulator r=40')

    # Visualize peak radius

    axes[1][2].imshow(accumulator[:,:,-1], cmap='gray')
    axes[1][2].set_title('Accumulator r=max')

    for row in axes: 
        for ax in row: 
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 70)
    print("Parameter Analysis:")
    print("  - Canny thresholds affect edge quality and thus detection")
    print("  - The radius parameters will allow us to focus on the radius that interest us, also the r_ssz is really important\n" \
    "because a really big one may lead us to skip some radius and therefore skip a potentially interesting circle for us.")
    print(" - Since we are rounding a bit, a circle could be detected under different radiuses. The parameter min_dist" \
    "ensures us that  we don't over detect the circles.")
    print(" - The angle step allows us to determine how we draw the circle around each point. This parameter impacts greatly the" \
    "performance of our program since we will iterate over it every time. Nonetheless is important to have it high enough to" \
    "guarantee a dense sampling process.")
    print("=" * 70)
    print("Task 2 complete!")
    print("=" * 70)


    # =============================================================
    print("=" * 70)
    print("Task 3: Mean Shift for Peak Detection in Hough Accumulator")
    print("=" * 70)

    bandwidth = 3
    print("Applying mean shift to find peaks...")
    peaks = myMeanShift(accumulator, bandwidth, threshold)
    print(f"Found {len(peaks)} raw peaks")
    
    # Build the raidus value corresponding to the raidus index
    usable_radius = np.linspace(
    min_radius,
    max_radius,
    int(np.floor((max_radius - min_radius) / r_ssz)))
    
    # Sorting peakes by vote strength
    # This way, we keep the strongest detections before weaker duplicates
    peaks.sort(key=lambda p: p[3], reverse=True)
    filtered = []
    
    # If two peaks are closer than this we consider them to be same circle
    # 15px is chosen for this image scale, but can be tuned
    min_center2 = 15**2
    # Simple non maximum suppression on the detected circles
    # We iterate through the peaks and keep only those peaks if it is not too close the previous one
    for (x, y, r, v) in peaks:
        keep = True
        # Compare this peak center to previously accepted center locations
        for (xx, yy, rr, vv) in filtered:
            # Center too close -> duplicate -> skip
            if (x-xx)**2 + (y-yy)**2 < min_center2:
                keep = False
                break
        if keep:
            filtered.append((x, y, r, v))

    peaks = filtered
    # Visualize corresponding circles on original image    
    vis = drawCirclesOnImage(img, peaks, radii_bins=usable_radius)
    
    plt.figure(figsize=(7,7))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected circles with Mean Shift (bandwidth={bandwidth}, peaks={len(peaks)})")
    plt.axis('off')
    plt.show()

    
    print("\n" + "=" * 70)
    print("Bandwidth Parameter Analysis:")
    # ...more analysis can be added here
    print("=" * 70)
    print("Task 3 complete!")
    

if __name__ == "__main__":
    main()
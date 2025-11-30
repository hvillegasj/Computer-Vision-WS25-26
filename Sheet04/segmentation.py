import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def average_image(img, segmented_mask):
    """
    Averages the colors on each mask component
    Args:
        img: Original image (h,w,3)
        segmented_mask: Segmentation mask from the image (h,w)

    Output:
        average_color_image: Image with average color per cluster (h,w,3)
    """
    average_color_image = np.zeros_like(img)

    colors = np.unique(segmented_mask)    

    for color in colors:
        mask = (segmented_mask == color)

        mean_color = np.mean(img[mask], axis=0)

        average_color_image[mask,:] = mean_color

    return average_color_image

def get_centers(segmented_mask, average_mask):
    """
    Subrutine that gets the centers of the clusters, with their color and position.
    For the color takes the average of the pixels of the cluster
    Args:
        segmented_mask: Mask with clustered superpixels to compute upon.
        average_mask: Mask with the average color per cluster
    Output:
        representative_vectors: Vector that stores one representative vector per cluster
    """
    clusters = np.unique(segmented_mask)

    #Get a representative feature vector from each cluster
    representative_vectors = []

    for cluster in clusters:
        #Take the mean coordinates among the coordinates of the cluster
        coords = np.argwhere(segmented_mask == cluster)
        centroid_y, centroid_x = np.mean(coords, axis=0)
        coord_y = np.round(centroid_y).astype(int)
        coord_x = np.round(centroid_x).astype(int)

        color_vector = average_mask[coord_y, coord_x]
        position_vector = np.array([coord_y, coord_x])
        cluster_rep_vector = np.concatenate([color_vector, position_vector])
        
        representative_vectors.append(cluster_rep_vector)

    return np.array(representative_vectors)

def superpixel_segmentation_mask(img, superpixel_mask, average_color_mask, K):
    """
    Function that merges the superpixel clusters
    Args:
        img: Source image
        superpixel_mask: Mask with the presegmentation by superpixels
        average_color_mask: Mask with the presegmentation by superpixels with averaged colors
        K: Amount of clusters to get after applying k-means
    Output:
        binary_mask: Binary mask for buildings and background
    """
    # Getting a feature vector per superpixel
    rep_vectors = get_centers(superpixel_mask, average_color_mask)
    color_features = rep_vectors[:, :3].astype(np.float32)
    
    # K-means clustering of superpixels
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    attempts = 5
    flags = cv2.KMEANS_RANDOM_CENTERS

    compactness, labels, centers = cv2.kmeans(
        color_features,
        K,
        None,
        criteria,
        attempts,
        flags
    )
    # labels shape: (N_superpixels, 1)
    superpixel_cluster_labels = labels.flatten().astype(np.uint8)
    
    # Mapping cluster labels to image grid
    seg_mask = np.zeros_like(superpixel_mask, dtype=np.uint8)
    for superpixel_id, cluster_label in enumerate(superpixel_cluster_labels):
        seg_mask[superpixel_mask == superpixel_id] = cluster_label
        
    # Deciding which cluster is "building" (i.e choosing the cluster with higher mean intensity)
    labels = np.unique(seg_mask)
    building_clusters = []
    stats = []
    
    for c in labels:
        mask_c = (seg_mask == c)
        if mask_c.sum() == 0:
            continue
        
        rgb_mean = img[mask_c].mean(axis=0)
        R, G, B = rgb_mean
        brightness = (R + G + B)/3.0
        spread = np.std(rgb_mean) # how different are R,G,B
        rg_ratio = R / max(G, 1e-6)
        rb_ratio = R / max(B, 1e-6)
        
        stats.append((c, brightness, spread, rg_ratio, rb_ratio))
    
    # buildings: bright, color-neutral, not extremely red
    for c, bright, spread, rg, rb in stats:
        if bright > 110 and spread < 30:
            building_clusters.append(c)
    # If nothign matched, take cluster with best (brightness - alpaha * spread)
    if not building_clusters:
        alpha = 1.5
        scores = [(c, bright - alpha*spread) for c, bright, spread, _, _ in stats]
        best = max(scores, key=lambda x: x[1])[0]
        building_clusters = [best]

    
    # Binary mask of all selected building clusters
    binary_mask = np.isin(seg_mask, building_clusters).astype(np.uint8)
    
    return binary_mask



def main():
    # Import the image of the UAV
    img_path = 'data/img_mosaic.tif'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    #Use the slic algorithm to get superpixels out of the image
    sp_mask = skimage.segmentation.slic(img, n_segments=400, compactness=18, start_label=0)

    average_color_mask = average_image(img, sp_mask)

    binary_mask = superpixel_segmentation_mask(img, sp_mask, average_color_mask, K=4)
    
    boundaries = skimage.segmentation.mark_boundaries(
        img,            # original RGB image
        binary_mask,    # your 0/1 mask as "labels"
        color=(0, 0, 0) # black boundaries;
    )
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title('Original image')
    
    axs[1].imshow(binary_mask, cmap='gray')
    axs[1].axis('off')
    axs[1].set_title('Segmentation mask')
    
    axs[2].imshow(boundaries)
    axs[2].axis('off')
    axs[2].set_title('Mask boundaries')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
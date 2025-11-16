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

def merge_clusters(segmented_mask, average_mask):
    """
    Merge close by clusters
    
    Args:
        segmented_mask: Mask with clustered superpixels to compute upon.
        average_mask: Mask with the average color per cluster
    """
    representative_vectors = get_centers(segmented_mask, average_mask)
    n = representative_vectors.shape[0]

    color_diff = representative_vectors[:, None, :3] - representative_vectors[None, :, :3]
    space_diff = representative_vectors[:, None, 3:] - representative_vectors[None, :, 3:]

    color_distance = np.sum(color_diff**2, axis = 2)
    space_distance = np.sum(space_diff**2, axis = 2)

    #We need to add an arbitrary amount in the diagonal since by construction all of these entries would be 0
    proximity_matrix = color_distance + space_distance + 500*np.identity(n)

    cluster_a, cluster_b = np.unravel_index(np.argmin(proximity_matrix), proximity_matrix.shape) 
    print (f'The closest clusters are cluster {cluster_a} and cluster {cluster_b}')
    pass

# Import the image of the UAV
img_path = 'Sheet04\data\img_mosaic.tif'
img=cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_RGB2BGR)

#Use the slic algorithm to get superpixels out of the image
sp_mask = skimage.segmentation.slic(img, 350, 18)

average_color_mask = average_image(img, sp_mask)

marked_img = skimage.segmentation.mark_boundaries(average_color_mask, sp_mask)

merge_clusters(sp_mask, average_color_mask)

# fig, axs = plt.subplots(1,3)

# axs[0].imshow(img)
# axs[0].axis('off')
# axs[0].set_title('Original image')

# axs[1].imshow(average_color_mask)
# axs[1].axis('off')
# axs[1].set_title('Averaged superpixels')

# axs[2].imshow(marked_img)
# axs[2].axis('off')
# axs[2].set_title('Segmentation result')

plt.figure()
plt.imshow(marked_img)
plt.axis('off')

plt.tight_layout()
plt.show()
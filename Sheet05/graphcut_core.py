import cv2
import numpy as np
import maxflow
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# TODO: Smooth the image

def populate_training_data(origin_img, label_img):
    """
    Fills an array with the pixels of the original image in the positions that the label indicates.
    Args:
        origin_img: Original image
        label_img: Label mask to design the pixels that are to be used
    Output:
        Flattened vector of the pixels in interest
    """
    non_zero_mask = (label_img != 0)
    non_zero_mask = np.any(non_zero_mask, axis = -1)
   
    return origin_img[non_zero_mask]

def main():
    #Import the images
    img_path = 'Sheet05\dataset\images\\bike_2007_005878.jpg'
    labels_path = 'Sheet05\dataset\images-labels\\bike_2007_005878-anno.png'

    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")

    if not os.path.exists(labels_path):
        print(f"Error: {labels_path} not found!")

    img = cv2.imread(img_path)
    labels = cv2.imread(labels_path)

    #Fill the histograms with the anotated pixels
    user_input_pixels = populate_training_data(img, labels)

    #Train the model that we will use to get our unitary weights
    #only 2 gaussians are used since we want to differenciate background from objects
    unitary_weights_model = GaussianMixture(n_components = 2).fit(user_input_pixels)

    #Apply the model to get the unitary energy for each pixel
    #After this, the unitary_weights_mask has 2 chanels [0] is the prob that the pixel is bg,
    # [1] is the prob that the object is from the object.
    flat_img = img.reshape((-1, 3))
    unitary_weights_mask = unitary_weights_model.predict_proba(flat_img)

    # WARNING: Code only to show, delete if is not helpful (i was just testing smth)
    # show_uw = unitary_weights_mask.reshape((img.shape[:2]))
    # plt.figure()
    # plt.imshow(show_uw, cmap='gray') 
    # plt.title("Estimation using only GMM")
    # plt.show()
    # print(unitary_weights_mask.shape)

    
    
if __name__ == "__main__":
    main()
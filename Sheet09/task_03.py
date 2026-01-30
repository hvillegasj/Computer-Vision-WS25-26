import numpy as np
import cv2
import random

np.set_printoptions(suppress=True)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_Homography_RANSAC(good_matches, kp_1, kp_2, image_1, image_2):
    # --- YOUR CODE HERE ---#
    # TODO compute best H transformation using RANSAC algorithm

    return best_H


def get_best_matches(des_1, des_2, thr=0.3):
    # --- YOUR CODE HERE ---#
    # TODO get best matches

    return good_matches


def stitch_multiple_images(images, homographies):
    # --- YOUR CODE HERE ---#
    # TODO Stitch multiple images given their homographies
    return result


def task_03():
    # Load all images
    image_paths = [r'./data/Fuji_1.png', r'./data/Fuji_2.png', r'./data/Fuji_3.png']
    # --- YOUR CODE HERE ---#
    # TODO read the images and extract key-points

    # --- YOUR CODE HERE ---#
    # TODO Compute matches between all pairs and visualize them
    # e.g
    for i in range(num_imgs):
        for j in range(i + 1, num_imgs):
            print(f"Processing pair {i} - {j}")
            # Match
            good_matches = get_best_matches(...)

            # TODO Visualize matches

    # --- YOUR CODE HERE ---#
    # TODO Find optimal reference image based on the computed matches

    # --- YOUR CODE HERE ---#
    # TODO Compute homographies to reference via RANSAC
    homographies = []

    # --- YOUR CODE HERE ---#
    # TODO visualize and save the final panorama
    print("Creating final panorama...")
    final_panorama = stitch_multiple_images(images, homographies)


if __name__ == "__main__":
    task_03()
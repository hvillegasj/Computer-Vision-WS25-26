import argparse
import cv2
import numpy as np
import math

np.set_printoptions(suppress=True, precision=5)


def click_event(event, x, y, flags, param):
    # --- YOUR CODE HERE ---#
    # TODO extract x, y coordinates from a mouse event (click)

def pick_points(image, window_name='image'):
    """
    Generic point picker.
    """
    img = image.copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, click_event)

    # --- YOUR CODE HERE ---#
    # TODO get the coordinates from a mouse event (click) for n_points and return the results in an array [n_points, 2]
    # TODO mark the selected points on the image

    return pts_xy

def compute_Perspective(pts_src, pts_target):
    # --- YOUR CODE HERE ---#
    # TODO compute Perspective transformation from a set of corresponding points
    # Do not use RANSAC here

    return


def compute_error(H, pts_src, pts_dst):
    # --- YOUR CODE HERE ---#
    # TODO Compute the errors between the transferred and measured points and print the alignment errors.

    return


def task_02(args):
    # Load images
    imgA = cv2.imread(args.a)
    imgB = cv2.imread(args.b)
    imgC = cv2.imread(args.c)

    # --- YOUR CODE HERE ---#
    # TODO Step 1: pick points correspondences for image pairs AB and BC
    # For the submission actually provide your selected points as hard-coded arrays and comment out the interactive selection.

    # --- YOUR CODE HERE ---#
    # TODO Step 2: Compute perspective transformation between A and B, print errors and visualize warped image AB.

    # --- YOUR CODE HERE ---#
    # TODO Step 3: Compute perspective transformation between B and C.

    # --- YOUR CODE HERE ---#
    # TODO Step 4: Derive A->C transformation and visualize stitched panorama ABC.



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=str, default="./data/A.png", help="Image A path")
    parser.add_argument("--b", type=str, default="./data/B.png", help="Image B path")
    parser.add_argument("--c", type=str, default="./data/C.png", help="Image C path")
    args = parser.parse_args()
    task_02(args)

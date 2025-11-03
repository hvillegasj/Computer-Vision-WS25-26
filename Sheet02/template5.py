# Template for Exercise 5 – Canny Edge Detector

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def gaussian_smoothing(img, sigma):
    """
    Apply Gaussian smoothing to reduce noise.
    """
    #We smooth the image using a gaussian filter which changes in size based on the sigma given.
    smoothed_img=cv2.GaussianBlur(img,(0,0),sigma)
    
    return smoothed_img


def compute_gradients(img):
    """
    Compute gradient magnitude and direction (Sobel-based).
    Return gradient_magnitude, gradient_angle.
    """
    #We use cv2.Sobel to calculate the partial derivatives of the image.
    horizontal_derivative=cv2.Sobel(img,cv2.CV_64F,1,0)
    vertical_derivative=cv2.Sobel(img,cv2.CV_64F,0,1)
    
    #We use numpy to compute the angles and the magnitude
    #Note: There are approximated ways to do it faster but we chose the precise approach using np.
    gradient_magnitude=np.hypot(horizontal_derivative,vertical_derivative)

    safe_division = np.divide(
    vertical_derivative.astype(float),
    horizontal_derivative.astype(float),
    out=np.zeros_like(vertical_derivative, dtype=float),
    where=horizontal_derivative != 0
    ) #Safe division to avoid dividing by 0 when calculating the angle.

    gradient_angle=np.arctan(safe_division)
    gradient_angle[horizontal_derivative == 0] = np.pi / 2 * np.sign(vertical_derivative[horizontal_derivative == 0])

    return gradient_magnitude, gradient_angle


def nonmax_suppression(mag, ang):
    """
    Perform non-maximum suppression to thin edges.
    """
    #Get the shape of the magnitude matrix
    size=np.shape(mag)

    mag_copy=mag.copy()

    for i in range(1,size[0]-1):
        for j in range(1,size[1]-1): #Iterate over the pixels in the magnitude matrix
            theta=ang[i,j]
            comparing_value_1=0.0
            comparing_value_2=0.0
                
            #We first clasify in which section of the circle is our angle and find the comparing values.
            if -np.pi/2 <= theta and theta < -np.pi/4:
                    #Since is pointing downwards we check if the pixels
                    # needed for the comparison are inbounds and then we
                    #  perform our interpolation and comparison.
                w= 1/np.abs(np.tan(theta))
                
                #Get our comparing values using a linear interpolation
                comparing_value_1=w*mag[i,j+1]+(1-w)*mag[i+1,j+1]
                comparing_value_2=w*mag[i,j-1]+(1-w)*mag[i-1,j-1]
                        
            elif -np.pi/4 <= theta and theta < 0:
                    #Since is pointing rightwards we check if the pixels
                    # needed for the comparison are inbounds and then we
                    #  perform our interpolation and comparison.
                w=np.abs(np.tan(theta))

                #Get our comparing values using a linear interpolation
                comparing_value_1=w*mag[i+1,j]+(1-w)*mag[i+1,j+1]
                comparing_value_2=w*mag[i-1,j]+(1-w)*mag[i-1,j-1]

            elif 0 <= theta and theta<np.pi/4:
                    #Since is pointing rightwards we check if the pixels
                    # needed for the comparison are inbounds and then we
                    #  perform our interpolation and comparison.
                w=np.abs(np.tan(theta))
                            
                #Get our comparing values using a linear interpolation
                comparing_value_1=w*mag[i+1,j]+(1-w)*mag[i+1,j-1]
                comparing_value_2=w*mag[i-1,j]+(1-w)*mag[i-1,j+1]

            else:
                    #Since is pointing upwards we check if the pixels
                    # needed for the comparison are inbounds and then we
                    #  perform our interpolation and comparison.
                w=1/np.abs(np.tan(theta))

                #Get our comparing values using a linear interpolation
                comparing_value_1=w*mag[i,j-1]+(1-w)*mag[i+1,j-1]
                comparing_value_2=w*mag[i,j+1]+(1-w)*mag[i-1,j+1]
            
            #Perform our comparison and set the magnitude value to 0 in case it is not a local maximum
            if mag[i,j]<comparing_value_1 or mag[i,j]<comparing_value_2:
                    mag_copy[i,j]=0
    
    return mag_copy
        
    
def double_threshold(nms, low, high):
    """
    Apply double thresholding to classify strong, weak, and non-edges.
    Return thresholded edge map.
    """
    img_copy=np.zeros_like(nms)

    img_copy = np.where(nms>high,255,0)+np.where((nms<high) & (nms>low),100,0) 
    #Ignore everything below the lower threshold and mark as border everything above it
    
    return img_copy


def hysteresis(edge_map, weak, strong):
    """
    Perform edge tracking by hysteresis.
    Return final binary edge map.
    """
    # Identify strong and weak edges
    strong_edges = (edge_map >= strong).astype(np.uint8)
    weak_edges = ((edge_map >= weak) & (edge_map < strong)).astype(np.uint8)

    # Initialize final edges with strong edges
    final_edges = np.copy(strong_edges)

    # 8-connectivity kernel
    kernel = np.ones((3,3), dtype=np.uint8)

    # Iteratively grow strong edges by connecting weak edges
    while True:
        # Dilate strong edges
        dilated = cv2.dilate(final_edges, kernel, iterations=1)

        # Find weak pixels connected to strong edges
        connected = np.logical_and(dilated, weak_edges).astype(np.uint8)

        # Add new connections to final edges
        new_final = np.logical_or(final_edges, connected).astype(np.uint8)

        # Stop if no new edges were added
        if np.array_equal(new_final, final_edges):
            break

        final_edges = new_final

    # Scale to 0-255
    final_edges = final_edges * 255

    return final_edges.astype(np.uint8)


def compute_metrics(manual_edges, cv_edges):
    """
    Compute MAD, precision, recall, and F1-score between two binary edge maps.
    """
    #We first normalize the arrays
    manual_edges=cv2.normalize(manual_edges,None,0,1,cv2.NORM_MINMAX)
    cv_edges=cv2.normalize(cv_edges,None,0,1,cv2.NORM_MINMAX)

    MAD = np.mean(np.abs(manual_edges-cv_edges))

    size=np.shape(manual_edges)

    tp, fp, tn, fn= 0, 0, 0, 0

    #We go over the whole image to clasify between false/true negative/positive edges detected.
    for i in range(size[0]):
        for j in range(size[1]):
            if manual_edges[i,j]==1 and cv_edges[i,j]==1:
                tp+=1
            elif manual_edges[i,j]==1 and cv_edges[i,j]==0:
                fp+=1
            elif manual_edges[i,j]==0 and cv_edges[i,j]==0:
                tn+=1
            elif manual_edges[i,j]==0 and cv_edges[i,j]==1:
                fn+=1

    precision = tp / (tp+fp)

    recall = tp / (tp+fn)

    F1_score= tp / (tp + 0.5*(fp + fn))

    return MAD, precision, recall, F1_score


# ==========================================================

#1. Load the grayscale image 'bonn.jpg'
img=cv2.cvtColor(cv2.imread("data/bonn.jpg"),cv2.COLOR_BGR2GRAY)

#2. Smooth the image using your Gaussian 
smoothing_factor= 3

img=gaussian_smoothing(img,smoothing_factor)

#3. Compute gradients (magnitude and direction)
grad_magn,grad_angle=compute_gradients(img)

#4. Apply non-maximum suppression
maximum_suppresed_image= nonmax_suppression(grad_magn,grad_angle)

#5. Apply double threshold (choose suitable low/high values)
low_thr=0.22*255
high_thr=0.38*255

threshholded_image=double_threshold(maximum_suppresed_image,low_thr,high_thr)

#6. Perform hysteresis to obtain final edges
hysteresis_image=hysteresis(threshholded_image,low_thr,high_thr)

# TODO: 7. Compare your result with cv2.Canny using MAD and F1-score
cv2_edges=cv2.Canny(img,low_thr,high_thr)

MAD, precision, recall, F1_score = compute_metrics(hysteresis_image,cv2_edges)

print(f"The MAD score between the two images is {MAD:.3f}")
# print(f"The precision score between the two images is {precision:.3f}")
# print(f"The recall score between the two images is {recall:.3f}")
print(f"The F1_score score between the two images is {F1_score:.3f}")

#8. Display original image, your edges, and OpenCV edges
fig, axes = plt.subplots(1, 3, figsize=(10, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(hysteresis_image, cmap='gray')
axes[1].set_title("Custom Edges")
axes[1].axis('off')

axes[2].imshow(cv2_edges, cmap='gray')
axes[2].set_title("CV2 Edges")
axes[2].axis('off')

plt.tight_layout()
plt.show()
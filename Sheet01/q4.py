import cv2
import numpy as np

# We create the matrixes corresponding to the kernels
kernel1=np.array([
    [0.0113,0.0838,0.0113],
    [0.0838,0.6193,0.0838],
    [0.0113,0.0838,0.0113]
    ])
kernel2=np.array([
    [-0.8984,0.1472,1.1410],
    [-1.9075,0.1566,2.1359],
    [-0.8659,0.0573,1.0337]
    ])

#We call cv2.SVDecomp to do the decomposition and store the vectors and the singular values

w1, u1, vt1 = cv2.SVDecomp(kernel1)
w2, u2, vt2 = cv2.SVDecomp(kernel2)

# Load images here
original_img_color = cv2.imread("bonn.jpg")  # Load bonn.jpg
original_img_gray = cv2.cvtColor(original_img_color, cv2.COLOR_RGB2GRAY)   # Convert to grayscale
 
#In order for it to be separable we check the magnitude of the singular values

def Checkseparability (singularValues):
    """
        Given the singular values of a matrix determines if it is separable by checking if more than
        one of them is  non-zero.

        singular Values: Vector of singular values of the matrix.
    """
    if singularValues[1]!=0:
        print("The kernel is not separable")
    else: print("The kernel is separable")

#We check for the separability of the kernels using the previous method.
Checkseparability(w1)
Checkseparability(w2)

#Create the approximation of the kernels by considering only the part corresponding to the greatest singular value.

approx_kernel1= w1[0]*np.outer(u1[0],vt1[0])
approx_kernel2= w2[0]*np.outer(u1[0],vt1[0])

def CompareKernels (image, original_kernel, approx_kernel,display=False):
    """
    Filters and displays the image using both the original and the approximated kernel
    then calculates the absolute pixel-wise difference and prints the maxium pixel error.

    image: Input image to filter.
    original_kernel: Full kernel.
    approx_kernel: Separable approximation of the kernel
    """
    #Calculate the filters using cv2.filter2D
    original_filtering=cv2.filter2D(image, -1,original_kernel)
    approx_filtering=cv2.filter2D(image,-1,approx_kernel)

    #Show the images to do a visual comparison if desired
    if(display):
        
        image_comparison=cv2.hconcat([original_filtering,approx_filtering])
        cv2.imshow("Original filter vs Approx filter",image_comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #Calculate the absolute pixel-wise difference between the filters
    diff_matrix=np.abs(original_filtering,approx_filtering)
    print(f"The maximum pixel wise difference between the filtered images is {np.max(diff_matrix)}")

#Run the compare functions with both kernels
CompareKernels(original_img_gray,kernel1,approx_kernel1)
CompareKernels(original_img_gray,kernel2,approx_kernel2)
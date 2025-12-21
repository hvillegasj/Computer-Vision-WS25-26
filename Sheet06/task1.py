import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import multivariate_normal

'''
BG_pivot is the same shape as the input image but with single channel, all pixels have value 1.
'''
def calculate_normal (x, mu, sigmaSQ):
    """
    Calculates probability of x in under the distribution N(mu,sigma).
    Assumes a 3-dimensional normal with diagonal covariance matrix.
    """
    eps = 1e-6  # numerical stability
    sigmaSQ = np.maximum(sigmaSQ, eps)

    diff = x - mu
    exponent = -0.5 * np.sum((diff**2) / sigmaSQ)

    denom = np.sqrt((2 * np.pi)**3 * np.prod(sigmaSQ))

    return np.exp(exponent) / denom


class MOG():
    def __init__(self,height=None, width=None, number_of_gaussians=None, background_thresh=None, lr=None):
        self.number_of_gaussians = number_of_gaussians
        self.background_thresh = background_thresh #The Gaussians with smaller variance correspond to the bg
        self.dist_thresh = 20
        self.lr = lr
        self.height = height
        self.width = width
        self.mus = np.zeros((self.height,self.width, self.number_of_gaussians,3)) ## assuming using color frames
        self.sigmaSQs = np.zeros((self.height, self.width, self.number_of_gaussians)) ## all color channels share the same sigma and covariance matrices are diagnalized
        self.omegas = np.zeros((self.height, self.width, self.number_of_gaussians)) 
        for i in range(self.height):
            for j in range(self.width):
                self.mus[i,j]=np.array([[122, 122, 122]]*self.number_of_gaussians) ##assuming a [0,255] color channel
                self.sigmaSQs[i,j]=[225.0] * self.number_of_gaussians #Diagonal covariance
                self.omegas[i,j]=[1.0 / self.number_of_gaussians] * self.number_of_gaussians
                
    def updateParam(self, img, BG_pivot): #finish this function
        for i in range(self.height):
            for j in range(self.width):
                #Test pixel value against each of its Gaussians
                pixel_color = img[i,j]
                distances = np.linalg.norm(pixel_color-self.mus[i,j], axis = 1)
                matches = (distances**2) < 9.0 * self.sigmaSQs[i,j]

                #Update the gaussians according to the matches
                if np.any(matches):
                    #Update and normalize the weights
                    self.omegas[i, j] = (1 - self.lr) * self.omegas[i, j] + self.lr * matches.astype(np.float32)
                    self.omegas[i,j] = self.omegas[i,j] / np.sum(self.omegas[i,j]) 

                    #Find the matched distribution and set up of the second learning rate and 
                    #match_index = np.argwhere(matches)
                    match_idx = np.where(matches)[0]
                    m = match_idx[np.argmin(distances[match_idx])]
                    rho = self.lr * calculate_normal(pixel_color, self.mus[i, j, m], self.sigmaSQs[i, j, m])
                    """match_probability = Calculate_normal(pixel_color, self.mus[i,j,match_index], self.sigmaSQs[i,j,match_index])
                    rho = self.lr * match_probability

                    # Update the mean of the matched distribution
                    old_mu = self.mus[i, j, match_index,:]
                    self.mus[i,j,match_index,:] = (1-rho) * old_mu + rho * pixel_color 

                    #Update the variance of the matched distribution
                    difference = pixel_color - self.mus[i,j,match_index]
                    diff_squared = np.dot(difference, difference)
                    self.sigmaSQs[i,j,match_index] = (1-rho) * self.sigmaSQs[i,j,match_index] + rho * diff_squared"""
                    
                    old_mu = self.mus[i, j, m].copy()
                    self.mus[i, j, m] = (1 - rho) * old_mu + rho * pixel_color
                    diff = pixel_color - self.mus[i, j, m]
                    diff_sq = float(np.sum(diff * diff))          # ||x - mu||^2
                    self.sigmaSQs[i, j, m] = (1 - rho) * self.sigmaSQs[i, j, m] + rho * diff_sq

                else:
                    """#Choose the distribution with the least amount of evidence
                    min_weight_index = np.argmin(self.omegas[i,j])

                    #Replace the distribution
                    self.mus[i,j,min_weight_index] = pixel_color 
                    self.sigmaSQs[i,j,min_weight_index] = 200.0
                    self.omegas[i,j,min_weight_index] = 0.02 #TODO: Adjust these two numbers (specially the second)

                    #Update and normalize the weights
                    self.omegas[i,j] = (1-self.lr) * self.omegas[i,j] + self.lr * matches
                    self.omegas[i,j] = self.omegas[i,j] / np.sum(self.omegas[i,j]) 

                    #Set up of the second learning rate and find the matched distribution
                    match_index = np.argwhere(matches)
                    match_probability =  Calculate_normal(pixel_color, self.mus[i,j,match_index], self.sigmaSQs[i,j,match_index])
                    rho = self.lr * match_probability

                    #Update the mean of the matched distribution
                    old_mu = self.mus[i, j, match_index, :]
                    self.mus[i,j,match_index,:] = (1-rho) * old_mu + rho * pixel_color 

                    #Update the variance of the matched distribution
                    difference = pixel_color - self.mus[i,j,match_index]
                    diff_squared = np.dot(difference, difference)
                    self.sigmaSQs[i,j,match_index] = (1-rho) * self.sigmaSQs[i,j,match_index] + rho * diff_squared"""
                    # If nothing matched, matches is all False therefore it would try to update the matched gaussian which does not exist
                    k = int(np.argmin(self.omegas[i, j]))
                    self.mus[i, j, k] = pixel_color
                    self.sigmaSQs[i, j, k] = 100.0
                    self.omegas[i, j, k] = 0.02

                    # normalize weights (or re-decay then normalize)
                    self.omegas[i, j] = self.omegas[i, j] / np.sum(self.omegas[i, j])

                
                #Now that we updated the GMM on the pixel decide if it is currently background or foreground
                #Sort descending by the ratio omega/sigma
                """ratios = self.omegas/self.sigmaSQs
                descending_ratio_indexes = np.argsort(ratios)[::-1]
                self.mus[i,j] = self.mus[i,j][descending_ratio_indexes]
                self.sigmaSQs[i,j] = self.sigmaSQs[i,j][descending_ratio_indexes]
                self.omegas[i,j] = self.omegas[i,j][descending_ratio_indexes]
                matches = matches[descending_ratio_indexes]"""
                
                ratios = self.omegas[i, j] / (self.sigmaSQs[i, j] + 1e-6)
                order = np.argsort(ratios)[::-1]

                self.mus[i, j] = self.mus[i, j][order]
                self.sigmaSQs[i, j] = self.sigmaSQs[i, j][order]
                self.omegas[i, j] = self.omegas[i, j][order]
                matches = matches[order]


                accumulative_sum = np.cumsum(self.omegas[i,j])
                #B = np.argmax(accumulative_sum > self.background_thresh) #Separates between background and foreground distributions
                B = np.searchsorted(accumulative_sum, self.background_thresh) + 1 # If the first componenet already crosses the threshold, it would return 0

                #If the pixel matches with a background distribution is background and viceversa

                if np.any(matches) and (np.argmax(matches) < B):
                    BG_pivot[i, j] = 0
                else:
                    BG_pivot[i, j] = 255

        return BG_pivot
    
img_1 = cv2.imread('imgs/0001.jpg')
h, w = img_1.shape[:2]
mog =  MOG(height=h, width=w, number_of_gaussians=3, background_thresh=0.5, lr=0.01)
              
for i in range(1, 4):#display first 3 labeled foreground images
    img = cv2.imread('imgs/{:04d}.jpg'.format(i))
    
    BG_pivot = np.zeros(img.shape[:2], dtype=np.uint8)   # new output buffer for this frame
    fg_mask = mog.updateParam(img, BG_pivot)
     
    cv2.imwrite('label{:04d}.jpg'.format(i), fg_mask)
    
# Fixed the code here because if the Mixture of Gaussian model does not carry the learned parameters across frames it never learns
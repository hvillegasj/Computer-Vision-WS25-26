import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import multivariate_normal

'''
BG_pivot is the same shape as the input image but with single channel, all pixels have value 1.
'''
def compute_rho (x, mu, sigmaSQ, lr):
    """
    Computes a dynamic learning rate rho for the matched Gaussian.
    """
    # avoid division by zero
    if sigmaSQ < 1e-6:
        sigmaSQ = 1e-6

    # squared distance between pixel and mean
    diff = x - mu
    dist_sq = np.sum(diff * diff)

    # confidence: high if x is close to mu, low otherwise
    confidence = np.exp(-0.5 * dist_sq / sigmaSQ)

    # scale with learning rate
    rho = lr * confidence

    # clamp rho to keep updates stable
    rho = np.clip(rho, 1e-4, 0.5 * lr)

    return rho

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
                    # If the pixel is close to Gaussian mean, rho is large and if it is far away i rho is small
                    rho = compute_rho(pixel_color, self.mus[i, j, m], self.sigmaSQs[i, j, m], self.lr)
                    
                    old_mu = self.mus[i, j, m].copy()
                    # Update mean
                    self.mus[i, j, m] = (1 - rho) * old_mu + rho * pixel_color
                    # Update variance using old mean
                    diff = pixel_color - old_mu
                    diff_sq = np.sum(diff * diff)

                    self.sigmaSQs[i, j, m] = (1 - rho) * self.sigmaSQs[i, j, m] + rho * diff_sq

                    # prevent variance from becoming too small
                    self.sigmaSQs[i, j, m] = max(self.sigmaSQs[i, j, m], 15.0)

                else:
                    # If nothing matched, matches is all False therefore it would try to update the matched gaussian which does not exist
                    k = int(np.argmin(self.omegas[i, j]))
                    self.mus[i, j, k] = pixel_color
                    self.sigmaSQs[i, j, k] = 100.0
                    self.omegas[i, j, k] = 0.02

                    # normalize weights (or re-decay then normalize)
                    self.omegas[i, j] = self.omegas[i, j] / np.sum(self.omegas[i, j])

                
                #Now that we updated the GMM on the pixel decide if it is currently background or foreground
                #Sort descending by the ratio omega/sigma
                
                ratios = self.omegas[i, j] / (self.sigmaSQs[i, j] + 1e-6)
                order = np.argsort(ratios)[::-1]

                self.mus[i, j] = self.mus[i, j][order]
                self.sigmaSQs[i, j] = self.sigmaSQs[i, j][order]
                self.omegas[i, j] = self.omegas[i, j][order]


                accumulative_sum = np.cumsum(self.omegas[i,j])
                #B = np.argmax(accumulative_sum > self.background_thresh) #Separates between background and foreground distributions
                B = np.searchsorted(accumulative_sum, self.background_thresh) + 1 # If the first componenet already crosses the threshold, it would return 0

                #If the pixel matches with a background distribution is background and viceversa
                # recompute matches with sorted components
                distances_sorted = np.linalg.norm(pixel_color - self.mus[i, j], axis=1)
                matches_sorted = (distances_sorted**2) < 9.0 * self.sigmaSQs[i, j]
                
                if np.any(matches_sorted) and (np.argmax(matches_sorted) < B):
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
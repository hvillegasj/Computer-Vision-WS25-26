import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import multivariate_normal

'''
BG_pivot is the same shape as the input image but with single channel, all pixels have value 1.
'''
def Calculate_normal (x, mu, sigma):
    """
    Calculates probability of x in under the distribution N(mu,sigma).
    Assumes a 3-dimensional normal.
    """
    denominator = (2*np.pi)**(3/2)*np.sqrt(np.linalg.det(sigma))
    numerator = np.exp(-0.5 * np.transpose(x-mu) @ np.inv(sigma) @ (x-mu))
    
    return numerator/denominator


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
                self.sigmaSQs[i,j]=[36.0] * self.number_of_gaussians
                self.omegas[i,j]=[1.0 / self.number_of_gaussians] * self.number_of_gaussians
                
    def updateParam(self, img, BG_pivot): #finish this function
        for i in range(self.height):
            for j in range(self.width):
                #Test pixel value against each of its Gaussians
                pixel_color = img[i,j]
                distances = np.linalg.norm(pixel_color-self.mus[i,j], axis = 1)
                if np.any(distances < 2.5 * self.sigmaSQs[i,j]):
                    #Update and normalize the weights
                    self.omegas[i,j] = (1-self.lr) * self.omegas[i,j] + self.lr * (distances < 2.5 * self.sigmaSQs[i,j])
                    self.omegas[i,j] = self.omegas[i,j] / np.sum(self.omegas[i,j]) 

                    #Set up of the second learning rate and find the matched distribution
                    match_index = np.argwhere(distances < 2.5 * self.sigmaSQs[i,j])
                    match_probability =  Calculate_normal(pixel_color, self.mus[i,j,match_index], self.sigmaSQs[i,j,match_index])
                    rho = self.lr * match_probability

                    #Update the mean of the matched distribution
                    self.mus[i,j,match_index] = (1-rho) * self.mus[i,j,match_index] + rho * pixel_color 

                    #Update the variance of the matched distribution
                    difference = pixel_color - self.mus[i,j,match_index]
                    self.sigmaSQs[i,j,match_index] = (1-rho) * self.mus[i,j,match_index] + rho * np.transpose(difference) * difference
                else:
                    #Choose the distribution with the lesser probability
                    probabilities =  Calculate_normal(pixel_color, self.mus[i,j], self.sigmaSQs[i,j])
                    min_prob_index = np.argmin(probabilities)

                    #Replace the distribution
                    self.mus[i,j,min_prob_index] = pixel_color 
                    self.sigmaSQs[i,j,min_prob_index] = 36.0
                    self.omegas[i,j,min_prob_index] = 4 #TODO: Adjust these two numbers (specially the second)

                    #Update and normalize the weights
                    self.omegas[i,j] = (1-self.lr) * self.omegas[i,j] + self.lr * (distances < 2.5 * self.sigmaSQs[i,j])
                    self.omegas[i,j] = self.omegas[i,j] / np.sum(self.omegas[i,j]) 

                    #Set up of the second learning rate and find the matched distribution
                    match_index = np.argwhere(distances < 2.5 * self.sigmaSQs[i,j])
                    match_probability =  Calculate_normal(pixel_color, self.mus[i,j,match_index], self.sigmaSQs[i,j,match_index])
                    rho = self.lr * match_probability

                    #Update the mean of the matched distribution
                    self.mus[i,j,match_index] = (1-rho) * self.mus[i,j,match_index] + rho * pixel_color 

                    #Update the variance of the matched distribution
                    difference = pixel_color - self.mus[i,j,match_index]
                    self.sigmaSQs[i,j,match_index] = (1-rho) * self.mus[i,j,match_index] + rho * np.transpose(difference) * difference

#TODO: Threshold the bg by the values omega/sigma and print the image
                
     
for i in range(1, 3+1):#display first 3 labeled foreground images
    img = cv2.imread('imgs/{:04d}.jpg'.format(i))
    h, w = img.shape[:2]
    mog = MOG(height = h, width = w, number_of_gaussians = 3, background_thresh = 0.5, lr = 0.01)
    label_img = mog.updateParam(img, np.ones(img.shape[:2]))
    cv2.imwrite('label{:04d}.jpg'.format(i), label_img)


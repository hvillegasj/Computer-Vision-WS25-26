# Template for Exercise 2 –  Fourier Transform and Image Reconstruction
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_fft(img):
    """
    Compute the Fourier Transform of an image and return:
    - The shifted spectrum
    - The magnitude
    - The phase
    """
    #We use numpy module for computing fast fourier transform to calculate the fourier transforms
    fourier_transform=np.fft.fft2(img)

    #We shift our Fourier transform matrix to get a centered transform.
    f_shifted = np.fft.fftshift(fourier_transform)

    #Matrix of magnitudes of our fourier transform
    fourier_transform_magnitude=np.abs(f_shifted)

    #Matrix of phases of our fourier transform
    fourier_transform_phase=np.angle(f_shifted)

    return f_shifted, fourier_transform_magnitude, fourier_transform_phase


def reconstruct_from_mag_phase(mag, phase):
    """
    Reconstruct an image from given magnitude and phase.
    """
    #Recovers the fourier transform of an image by considering the polar expresion of each entry.
    f_transform=mag*np.exp(1j*phase)

    #Since our inputs will be shifted we need to invert the shift.
    f_unshifted=np.fft.ifftshift(f_transform)

    img=np.fft.ifft2(f_unshifted)

    return np.real(img)


def compute_mad(a, b):
    """
    Compute the Mean Absolute Difference (MAD) between two images.
    """
    #We compute the matrix of differences
    absolute_difference=np.abs(a-b)

    #We take the mean among all the entries of the previous matrix and return it
    return np.mean(absolute_difference)

# ==========================================================
# 1. Load the two grayscale images (1.png and 2.png)
image_1=cv2.imread("data/1.png",cv2.IMREAD_GRAYSCALE)
image_2=cv2.imread("data/2.png",cv2.IMREAD_GRAYSCALE)

# 2. Compute magnitude and phase of both images
transform_img_1, mag_img_1, phase_img_1=compute_fft(image_1)
transform_img_2, mag_img_2, phase_img_2=compute_fft(image_2)

# 3. Swap magnitude and phase between the two images.
# 4. Reconstruct and save the swapped results
reconstruction_1=reconstruct_from_mag_phase(mag_img_2,phase_img_1).astype(np.uint8)
reconstruction_2=reconstruct_from_mag_phase(mag_img_1,phase_img_2).astype(np.uint8)

# TODO: 5. Compute and print the MAD values between originals and reconstructions
MAD_1=compute_mad(image_1,reconstruction_1)
MAD_2=compute_mad(image_2,reconstruction_2)

print(f'The MAD for the first image is {MAD_1:.2f}')
print(f'The MAD for the second image is {MAD_2:.2f}')

# TODO: 6. Visualize all images (originals, magnitude, phase, reconstructions)
#Before visualizing the magnitudes and the phases we normalize them by using a logarithm for
#range correction and normalize to format it before passing to the plot function.
mag_img_1_log = np.log(1 + mag_img_1)
mag_img_2_log = np.log(1 + mag_img_2)

mag_img_1_vis=cv2.normalize(mag_img_1_log,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
mag_img_2_vis=cv2.normalize(mag_img_2_log,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

phase_img_1=cv2.normalize(phase_img_1,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
phase_img_2=cv2.normalize(phase_img_2,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

plt.figure(figsize=(10, 4))

plt.subplot(2, 4, 1)
plt.imshow(image_1, cmap='gray')
plt.title("Photographer original")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(mag_img_1_vis, cmap='gray')
plt.title("Photographer magnitude")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(phase_img_1, cmap='gray')
plt.title("Photographer phase")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(reconstruction_1, cmap='gray')
plt.title("Photographer with Einstein's phase")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(image_2, cmap='gray')
plt.title("Einstein original")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(mag_img_2_vis, cmap='gray')
plt.title("Einstein's magnitude")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(phase_img_2, cmap='gray')
plt.title("Einstein's phase")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.imshow(reconstruction_2, cmap='gray')
plt.title("Einstein with photographer's phase")
plt.axis("off")

plt.tight_layout()
plt.show()

#Discussion:
#The presence of the tripod in the photography of the photographer causes very visible diagonal lines that
#can be evidenced on its magnitude diagram. In the other hand, Einstein shows clearly a tendency for vertical lines
#that originate thanks to the courtain in the back of the picture.

# When we mix the images we can kind of see how each picture patterns arise in the other (diagonals in Einstein's and
# squared in the photographers). Is interesting how even though these change, the phase still manages to keep the images
# recognizible enough.
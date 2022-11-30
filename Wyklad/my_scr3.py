# load a test image (with patterned noises)
import cv2
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread('ClownOrig.jpg', 0)

# transform to Fourier Domain

IM = np.fft.fft2(im)
IMs = np.fft.fftshift(IM)

# run it in Python for interactive demo

N = IMs.shape[0]
x, y = np.meshgrid(np.arange(N), np.arange(N))

# notch filter generation (need to understand)

a1 = 0.008
a2 = 0.008

NF1 = 1 - np.exp(-a1*(x-190)**2 - a2*(y-123)**2) # Gaussian
NF2 = 1 - np.exp(-a1*(x-104)**2 - a2*(y-172)**2) # Gaussian
NF3 = 1 - np.exp(-a1*(x-126)**2 - a2*(y-135)**2) # Gaussian
NF4 = 1 - np.exp(-a1*(x-168)**2 - a2*(y-161)**2) # Gaussian

Z = NF1*NF2*NF3*NF4
IMFs = IMs*Z

IMFr = np.fft.ifftshift(IMFs)
imfr = np.fft.ifft2(IMFr)

plt.figure(figsize = (12,12))

plt.subplot(2,2,1)
plt.title('Original Image')
plt.imshow(im, cmap = 'gray')
plt.axis('off')

plt.subplot(2,2,2)
plt.title('Filter in Freq')
plt.imshow(np.log(1+np.absolute(Z)), cmap = 'gray')
plt.axis('off')

plt.subplot(2,2,3)
plt.title('Filtered Image')
plt.imshow(np.real(imfr), cmap = 'gray')
plt.axis('off')

plt.subplot(2,2,4)
plt.title('Filtered FFT')
plt.imshow(np.log(1+np.absolute(IMFs)), cmap = 'gray')
plt.axis('off')

plt.show()
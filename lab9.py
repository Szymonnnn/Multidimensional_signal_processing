import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage.filters import threshold_otsu
from skimage import morphology
import skimage
from skimage.color import rgb2gray
from skimage.morphology import disk

img = plt.imread("fingerprint.jpg")
grayscale = rgb2gray(img)

binary = grayscale < 0.6

fig, ax = plt.subplots(4, 6, figsize=(12, 12))

footprint = disk(1)

ax[0, 0].imshow(footprint, cmap='binary')
ax[0, 0].set_title("footprint")

binary_erosion = morphology.binary_erosion(binary, footprint).astype(np.uint8)

ax[0, 1].imshow(binary_erosion, cmap='binary')
ax[0, 1].set_title("binary_erosion")

binary_dilation = morphology.binary_dilation(binary, footprint).astype(np.uint8)

ax[0, 2].imshow(binary_dilation, cmap='binary')
ax[0, 2].set_title("binary_dilation")

binary_opening = morphology.binary_opening(binary, footprint).astype(np.uint8)

ax[0, 3].imshow(binary_opening, cmap='binary')
ax[0, 3].set_title("binary_opening")

binary_closing = morphology.binary_closing(binary, footprint).astype(np.uint8)

ax[0, 4].imshow(binary_closing, cmap='binary')
ax[0, 4].set_title("binary_closing")


result = np.empty((binary.shape[0], binary.shape[1], 3)).astype(float)
result[:,:,0] = binary_dilation
result[:,:,1] = binary_opening
result[:,:,2] = binary_closing

ax[0, 5].imshow(result, cmap='binary')


footprint = disk(5)
ax[1, 0].imshow(footprint, cmap='binary')

binary_erosion = morphology.binary_erosion(binary, footprint).astype(np.uint8)
ax[1, 1].imshow(binary_erosion, cmap='binary')

binary_dilation = morphology.binary_dilation(binary, footprint).astype(np.uint8)
ax[1, 2].imshow(binary_dilation, cmap='binary')

binary_opening = morphology.binary_opening(binary, footprint).astype(np.uint8)
ax[1, 3].imshow(binary_opening, cmap='binary')

binary_closing = morphology.binary_closing(binary, footprint).astype(np.uint8)
ax[1, 4].imshow(binary_closing, cmap='binary')


result = np.empty((binary.shape[0], binary.shape[1], 3)).astype(float)
result[:,:,0] = binary_dilation
result[:,:,1] = binary_opening
result[:,:,2] = binary_closing

ax[1, 5].imshow(result, cmap='binary')


footprint = morphology.rectangle(nrows=10, ncols=1)
ax[2, 0].imshow(footprint, cmap='binary_r')

binary_erosion = morphology.binary_erosion(binary, footprint).astype(np.uint8)
ax[2, 1].imshow(binary_erosion, cmap='binary')

binary_dilation = morphology.binary_dilation(binary, footprint).astype(np.uint8)
ax[2, 2].imshow(binary_dilation, cmap='binary')

binary_opening = morphology.binary_opening(binary, footprint).astype(np.uint8)
ax[2, 3].imshow(binary_opening, cmap='binary')

binary_closing = morphology.binary_closing(binary, footprint).astype(np.uint8)
ax[2, 4].imshow(binary_closing, cmap='binary')


result = np.empty((binary.shape[0], binary.shape[1], 3)).astype(float)
result[:,:,0] = binary_dilation
result[:,:,1] = binary_opening
result[:,:,2] = binary_closing

ax[2, 5].imshow(result, cmap='binary')


footprint = morphology.rectangle(nrows=1, ncols=11)
ax[3, 0].imshow(footprint, cmap='binary_r')

binary_erosion = morphology.binary_erosion(binary, footprint).astype(np.uint8)
ax[3, 1].imshow(binary_erosion, cmap='binary')

binary_dilation = morphology.binary_dilation(binary, footprint).astype(np.uint8)
ax[3, 2].imshow(binary_dilation, cmap='binary')

binary_opening = morphology.binary_opening(binary, footprint).astype(np.uint8)
ax[3, 3].imshow(binary_opening, cmap='binary')

binary_closing = morphology.binary_closing(binary, footprint).astype(np.uint8)
ax[3, 4].imshow(binary_closing, cmap='binary')

result = np.empty((binary.shape[0], binary.shape[1], 3)).astype(float)
result[:,:,0] = binary_dilation
result[:,:,1] = binary_opening
result[:,:,2] = binary_closing

ax[3, 5].imshow(result, cmap='binary')

plt.savefig("lab9.png")
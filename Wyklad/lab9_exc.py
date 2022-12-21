import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage.filters import threshold_otsu
from skimage import morphology
import skimage
from skimage.color import rgb2gray
img = skimage.data.chelsea()
grayscale = rgb2gray(img)

fig, ax = plt.subplots(4, 2, figsize=(8, 12))

ax[0,0].imshow(img)
ax[0,0].set_title("Original")
ax[0,1].imshow(grayscale, cmap='binary')
ax[0,1].set_title("Grayscale")

wsp = 2-1
m_bit = np.rint(grayscale*wsp)
min_m_bit = np.min(m_bit)
max_m_bit = np.max(m_bit)
ax[1,0].imshow(m_bit, cmap = 'binary')
ax[1,0].set_title("min: " + str(min_m_bit) + ", max: " + str(max_m_bit))

thresh = threshold_otsu(grayscale)
binary = grayscale > thresh

ax[1,1].imshow(binary, cmap='binary')
ax[1,1].set_title("Tresholded")

S1 = np.array([[0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [1, 1, 1, 1, 1],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0]])

S2 = np.array([[1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1]])

edited = morphology.binary_erosion(binary, S1).astype(np.uint8)

ax[2,0].imshow(edited, cmap='binary')
ax[2,0].set_title("Erosion")

edited_2 = morphology.binary_erosion(edited, S1).astype(np.uint8)

ax[2,1].imshow(edited_2, cmap='binary')
ax[2,1].set_title("Erosion x2")

edited_3 = morphology.binary_dilation(edited_2, S2).astype(np.uint8)

ax[3,0].imshow(edited_3, cmap='binary')
ax[3,0].set_title("Dilatation")

edited_4 = morphology.binary_dilation(edited_3, S2).astype(np.uint8)

ax[3,1].imshow(edited_4, cmap='binary')
ax[3,1].set_title("Dilatation x2")

fig.tight_layout()
plt.savefig("Wyklad/lab9_exc.png")
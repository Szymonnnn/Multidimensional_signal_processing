from skimage.data import chelsea
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, watershed, quickshift
from skimage.color import rgb2gray
from skimage.color import label2rgb
from skimage.feature import canny

raw_image = chelsea()
meanImg = np.mean(raw_image, axis=2)

fig, ax = plt.subplots(3, 4, figsize = (32, 20))


segments1 = slic(image=raw_image)

ax[0,0].imshow(segments1, cmap='twilight')
ax[0,0].set_title("segments: " + str(len(np.unique(segments1))))
ax[0,0].set_ylabel('SLIC')

segments2 = watershed(image=meanImg)

ax[1,0].imshow(segments2, cmap='twilight')
ax[1,0].set_title("segments: " + str(len(np.unique(segments2))))
ax[1,0].set_ylabel('Watershed')

segments3 = quickshift(image=raw_image)

ax[2,0].imshow(segments3, cmap='twilight')
ax[2,0].set_title("segments: " + str(len(np.unique(segments3))))
ax[2,0].set_ylabel('Quickshift')


p01 = label2rgb(segments1, image=raw_image, kind='overlay')
ax[0,1].imshow(p01)

p11 = label2rgb(segments2, image=raw_image, kind='overlay')
ax[1,1].imshow(p11)

p21 = label2rgb(segments3, image=raw_image, kind='overlay')
ax[2,1].imshow(p21)


p02 = label2rgb(segments1, image=raw_image, kind='avg')
ax[0,2].imshow(p02)

p12 = label2rgb(segments2, image=raw_image, kind='avg')
ax[1,2].imshow(p12)

p22 = label2rgb(segments3, image=raw_image, kind='avg')
ax[2,2].imshow(p22)


p = canny(meanImg, sigma = 3.0)
p02[p]=1
ax[0,3].imshow(p02)

p12[p]=1
ax[1,3].imshow(p12)

p22[p]=1
ax[2,3].imshow(p22)

plt.savefig("lab10.png")
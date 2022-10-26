from skimage.data import chelsea
import numpy as np
import matplotlib.pyplot as plt

#przestrzeń na ploty
fig, ax = plt.subplots(2, 4, figsize = (12, 12/1.618))

#wczytujemy obraz
D = 8
L = np.power(2, D).astype(int)
raw_image = chelsea()

ax[0,0].imshow(raw_image)

#przygotujmy transformacje monochromatyczną ([r, g, b])
monochrome_transform = np.array([0, 1, 1])
monochrome_transform = monochrome_transform / np.sum(monochrome_transform)

#dokonujemy transformacji
#print(raw_image.shape, monochrome_transform[None, None])
mono_image = raw_image * monochrome_transform[None, None]
mono_image = np.sum(mono_image, axis=-1).astype(np.uint8)

ax[1, 0].imshow(mono_image, cmap='binary_r')

# histogram
hist = np.unique(mono_image, return_counts=True)
ax[1,1].scatter(*hist, c='black', marker= 'x')

# wektorowa forma histogramu
vhist = np.zeros((L))
vhist[hist[0]] = hist[1]
vhist /= np.sum(vhist)

# zaprezentujmy
ax[1, 2].plot(vhist, c='black')
ax[1, 2].set_ylim(0, .1)

# dystrybuanta
vdist = np.cumsum(vhist)
ax[1,3].plot(vdist, c='black')

plt.savefig("foo.png")
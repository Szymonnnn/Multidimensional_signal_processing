import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import radon, iradon, rescale

from skimage.data import shepp_logan_phantom
image = shepp_logan_phantom()

fig, ax = plt.subplots(4, 4, figsize = (12, 12))

theta = np.linspace(0, 180, 100)
ax[2, 0].imshow(image, cmap = 'binary_r')

sinogram = radon(image, theta=theta)
ax[2, 1].imshow(sinogram, cmap = 'binary_r', aspect='auto', interpolation='nearest')

rec_image = iradon(sinogram, theta=theta)
ax[2, 2].imshow(rec_image, cmap = 'binary_r')

err = image - rec_image
ax[2, 3].imshow(err, cmap = 'binary')

#zad 3

theta = np.linspace(0, 180, 10)
ax[0, 0].imshow(image, cmap = 'binary_r')

sinogram = radon(image, theta=theta)
ax[0, 1].imshow(sinogram, cmap = 'binary_r', aspect='auto', interpolation='nearest')

rec_image = iradon(sinogram, theta=theta)
ax[0, 2].imshow(rec_image, cmap = 'binary_r')

err = image - rec_image
ax[0, 3].imshow(err, cmap = 'binary')




theta = np.linspace(0, 180, 30)
ax[1, 0].imshow(image, cmap = 'binary_r')

sinogram = radon(image, theta=theta)
ax[1, 1].imshow(sinogram, cmap = 'binary_r', aspect='auto', interpolation='nearest')

rec_image = iradon(sinogram, theta=theta)
ax[1, 2].imshow(rec_image, cmap = 'binary_r')

err = image - rec_image
ax[1, 3].imshow(err, cmap = 'binary')




theta = np.linspace(0, 180, 1000)
ax[3, 0].imshow(image, cmap = 'binary_r')

sinogram = radon(image, theta=theta)
ax[3, 1].imshow(sinogram, cmap = 'binary_r', aspect='auto', interpolation='nearest')

rec_image = iradon(sinogram, theta=theta)
ax[3, 2].imshow(rec_image, cmap = 'binary_r')

err = image - rec_image
ax[3, 3].imshow(err, cmap = 'binary')


plt.savefig("lab7.png")

#cz2
image = image[50:350, 50:350]
fig2, ax = plt.subplots(4, 4, figsize = (12, 12))


theta = np.linspace(0, 180, 100)
ax[2, 0].imshow(image, cmap = 'binary_r')

sinogram = radon(image, theta=theta)
ax[2, 1].imshow(sinogram, cmap = 'binary_r', aspect='auto', interpolation='nearest')

rec_image = iradon(sinogram, theta=theta)
ax[2, 2].imshow(rec_image, cmap = 'binary_r')

err = image - rec_image
ax[2, 3].imshow(err, cmap = 'binary')

#zad 3

theta = np.linspace(0, 180, 10)
ax[0, 0].imshow(image, cmap = 'binary_r')

sinogram = radon(image, theta=theta)
ax[0, 1].imshow(sinogram, cmap = 'binary_r', aspect='auto', interpolation='nearest')

rec_image = iradon(sinogram, theta=theta)
ax[0, 2].imshow(rec_image, cmap = 'binary_r')

err = image - rec_image
ax[0, 3].imshow(err, cmap = 'binary')




theta = np.linspace(0, 180, 30)
ax[1, 0].imshow(image, cmap = 'binary_r')

sinogram = radon(image, theta=theta)
ax[1, 1].imshow(sinogram, cmap = 'binary_r', aspect='auto', interpolation='nearest')

rec_image = iradon(sinogram, theta=theta)
ax[1, 2].imshow(rec_image, cmap = 'binary_r')

err = image - rec_image
ax[1, 3].imshow(err, cmap = 'binary')




theta = np.linspace(0, 180, 1000)
ax[3, 0].imshow(image, cmap = 'binary_r')

sinogram = radon(image, theta=theta)
ax[3, 1].imshow(sinogram, cmap = 'binary_r', aspect='auto', interpolation='nearest')

rec_image = iradon(sinogram, theta=theta)
ax[3, 2].imshow(rec_image, cmap = 'binary_r')

err = image - rec_image
ax[3, 3].imshow(err, cmap = 'binary')

plt.savefig("lab7_2.png")
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.util import img_as_ubyte

img = plt.imread("image1.jpeg")
mono = np.mean(img, axis=2)

fig, ax = plt.subplots(3, 2, figsize = (12, 12))

ax[0, 0].imshow(mono, cmap = 'binary_r')


ft = np.fft.fft2(mono)
ft = np.fft.fftshift(ft)
abs = np.abs(ft)
log = np.log(abs)
ax[0, 1].imshow(log, cmap = 'binary_r')

min_log = np.min(log)
max_log = np.max(log)
norm = (log - min_log)
norm /= np.max(norm)

threshold = 0.5

binary=np.copy(norm)
binary[norm < threshold] = 0
binary[norm >= threshold] = 1

ax[1, 0].imshow(binary, cmap = 'binary_r')

threshold = 0.8

binary2=np.copy(norm)
binary2[norm < threshold] = 0
binary2[norm >= threshold] = 1

ax[1, 1].imshow(binary2, cmap = 'binary_r')

points = np.argwhere(binary2)
P1 = points[0]
P2 = points[1]
P3 = points[-2]
P4 = points[-1]

ft[75, :] = 0
ft[93, :] = 0
ft[131, :] = 0
ft[149, :] = 0


img_abs = np.abs(ft)
img_log = np.log(img_abs)
ax[2, 0].imshow(img_log, cmap = 'binary_r')

post_img = ft = np.fft.ifftshift(ft)
post_img = np.fft.ifft2(post_img).real

ax[2, 1].imshow(post_img, cmap = 'binary_r')

plt.savefig("lab6.png")


#ZAD 3


img2 = plt.imread("image2.jpg")

fig, ax = plt.subplots(2, 2, figsize = (12, 12))

ax[0, 0].imshow(img2, cmap = 'binary_r')

ft2 = np.fft.fft2(img2)
ft2 = np.fft.fftshift(ft2)
abs2 = np.abs(ft2)
log2 = np.log(abs2)
ax[0, 1].imshow(log2, cmap = 'binary_r')

rr, cc = skimage.draw.disk((298, 235), 10, shape=ft2.shape)
ft2[rr,cc]=0
rr, cc = skimage.draw.disk((170, 235), 10, shape=ft2.shape)
ft2[rr,cc]=0
rr, cc = skimage.draw.disk((298, 400), 10, shape=ft2.shape)
ft2[rr,cc]=0
rr, cc = skimage.draw.disk((170, 400), 10, shape=ft2.shape)
ft2[rr,cc]=0
rr, cc = skimage.draw.disk((230, 160), 10, shape=ft2.shape)
ft2[rr,cc]=0
rr, cc = skimage.draw.disk((230, 480), 10, shape=ft2.shape)
ft2[rr,cc]=0
rr, cc = skimage.draw.disk((365, 320), 10, shape=ft2.shape)
ft2[rr,cc]=0
rr, cc = skimage.draw.disk((110, 320), 10, shape=ft2.shape)
ft2[rr,cc]=0
abs2 = np.abs(ft2)
log2 = np.log(abs2)

ax[1, 0].imshow(log2, cmap = 'binary_r')

post_img2 = ft2 = np.fft.ifftshift(ft2)
post_img2 = np.fft.ifft2(post_img2).real

ax[1, 1].imshow(post_img2, cmap = 'binary_r')

plt.savefig("lab6_2.png")
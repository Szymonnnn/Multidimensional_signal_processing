import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(4, 2, figsize=(10, 13))
fig.subplots_adjust(hspace=.4)

import skimage
img = skimage.data.chelsea()
ax[0, 0].imshow(img)
ax[0, 0].set_title("original")

mono = np.mean(img, axis=2)[::8, ::8]
ax[0, 1].imshow(mono, cmap = 'binary_r')
ax[0, 1].set_title("mono")

from skimage.transform import AffineTransform, warp
angle = np.pi/12
matrix = np.matrix([[np.cos(angle), -np.sin(angle), 0], 
                    [np.sin(angle), np.cos(angle), 0], 
                    [0, 0, 1]])
transform = AffineTransform(matrix = matrix)
img_rot = warp(mono, transform.inverse)
ax[1, 0].imshow(img_rot, cmap = 'binary_r')
ax[1, 0].set_title("rotacja")


shear = 0.5
matrix = np.matrix([[1, shear, 0], 
                    [0, 1, 0], 
                    [0, 0, 1]])
transform = AffineTransform(matrix = matrix)
img_trans = warp(mono, transform.inverse)
ax[1, 1].imshow(img_trans, cmap = 'binary_r')
ax[1, 1].set_title("pochylenie")

from scipy import interpolate
new_x = np.linspace(0, img_rot.shape[1]*8, img_rot.shape[1])
new_y = np.linspace(0, img_rot.shape[0]*8, img_rot.shape[0])
print(new_x, new_y)
iterpol_rot = interpolate.interp2d(new_x, new_y, img_rot, kind='cubic')
ax[2, 0].imshow(iterpol_rot(np.linspace(0, img_rot.shape[1]*8, img_rot.shape[1]),
np.linspace(0, img_rot.shape[0]*8, img_rot.shape[0])), cmap = 'binary_r')
ax[2, 0].set_title("interpol cubic")

new_x = np.linspace(0, img_trans.shape[1]*8, img_trans.shape[1])
new_y = np.linspace(0, img_trans.shape[0]*8, img_trans.shape[0])
print(new_x, new_y)
iterpol_rot = interpolate.interp2d(new_x, new_y, img_trans, kind='cubic')
ax[2, 1].imshow(iterpol_rot(np.linspace(0, img_trans.shape[1]*8, img_trans.shape[1]*8),
np.linspace(0, img_trans.shape[0]*8, img_trans.shape[0]*8)), cmap = 'binary_r')
ax[2, 1].set_title("interpol cubic")
plt.show()

print(np.round(iterpol_rot(np.linspace(0, img_trans.shape[1]*8, img_trans.shape[1]),
np.linspace(0, img_trans.shape[0]*8, img_trans.shape[0])))[:15][:15])


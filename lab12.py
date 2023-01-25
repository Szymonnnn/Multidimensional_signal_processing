import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.transform import rescale

img = skimage.data.camera()
img_resc = rescale(img, 128/img.shape[0], anti_aliasing=False)

fig, ax = plt.subplots(2, 3, figsize = (12, 12))

ax[0, 0].imshow(img_resc, cmap = 'binary_r')


g_x = np.zeros((img_resc.shape[0], img_resc.shape[1]))
g_y = np.zeros((img_resc.shape[0], img_resc.shape[1]))

for x in range(img_resc.shape[0]):
    for y in range(img_resc.shape[1]):
        wsp_x1 = x+1
        wsp_x2 = x-1
        if ((wsp_x1>=0 and wsp_x1 < img_resc.shape[0]) and (wsp_x2>=0 and wsp_x2 < img_resc.shape[0])):
            x_ = img_resc[x+1,y]
            _x = img_resc[x-1, y]
            g_x[x, y] = x_ - _x
        else:
            g_x[x, y] = 0

ax[0, 1].imshow(g_x, cmap = 'binary_r')

for x in range(img_resc.shape[0]):
    for y in range(img_resc.shape[1]):
        wsp_y1 = y+1
        wsp_y2 = y-1
        if ((wsp_y1>=0 and wsp_y1 < img_resc.shape[0]) and (wsp_y2>=0 and wsp_y2 < img_resc.shape[0])):
            y_ = img_resc[x,y+1]
            _y = img_resc[x, y-1]
            g_y[x, y] = y_ - _y
        else:
            g_y[x, y] = 0

ax[0, 2].imshow(g_y, cmap = 'binary_r')

mag = np.sqrt(g_x*g_x + g_y*g_y)
ax[1, 0].imshow(mag, cmap = 'binary_r')

angle = np.arctan(g_y/g_x) + np.pi/2

ax[1, 1].imshow(angle, cmap = 'binary_r')
plt.savefig("lab12.png")

s = 8
count = 0

mask = np.zeros((img_resc.shape[0], img_resc.shape[1]))
i=0
for x in range(int(mask.shape[0]/s)):
    for y in range(int(mask.shape[1]/s)):
        mask[x*s : (x+1)*s, y*s : (y+1)*s] = count
        count += 1

fig, ax = plt.subplots(2, 2, figsize = (12, 12))
ax[0, 0].imshow(mask)

bins = 9

hog = np.zeros((int(img_resc.shape[0]/s), int(img_resc.shape[1]/s),bins))

step = np.pi/bins

for i in range(16*16):
    ang_v = angle[mask == i]
    mag_v = mag[mask == i]

    for i in range(bins):
        start = i*step
        end = (i+1)*step
        b_mask = np.zeros_like(ang_v)
        b_mask = ang_v[tu cos trzeba napisać]

plt.savefig("lab12_2.png")
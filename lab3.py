from skimage.data import chelsea
import numpy as np
import matplotlib.pyplot as plt

def hist(image):
    hist = np.unique(image[:,:,0], return_counts=True)
    vhist = np.zeros((L))
    vhist[hist[0]] = hist[1]
    vhist_r = vhist/ np.sum(vhist)

    hist = np.unique(image[:,:,1], return_counts=True)
    vhist = np.zeros((L))
    vhist[hist[0]] = hist[1]
    vhist_g = vhist/ np.sum(vhist)

    hist = np.unique(image[:,:,2], return_counts=True)
    vhist = np.zeros((L))
    vhist[hist[0]] = hist[1]
    vhist_b = vhist/ np.sum(vhist)

    return vhist_r, vhist_g, vhist_b

fig, ax = plt.subplots(6, 3, figsize = (12, 12))
raw_image = chelsea()

D = 8
L = np.power(2, D).astype(int)

lut_base = np.arange(0,L)
lut = np.linspace(0,255,L).astype(int)
lut_negation = np.linspace(255,0,L).astype(int)
lut_sin = np.sin(lut)

ax[0,0].plot(lut_base, lut)
image = lut[raw_image]
ax[0,1].imshow(image, cmap='binary_r')

hist_r, hist_g, hist_b = hist(raw_image)
ax[0, 2].plot(hist_r, c='red')
ax[0, 2].plot(hist_g, c='green')
ax[0, 2].plot(hist_b, c='blue')



ax[1,0].plot(lut_base, lut_negation)
negated_image = lut_negation[raw_image]
ax[1,1].imshow(negated_image, cmap='binary_r')

hist_r, hist_g, hist_b = hist(negated_image)
ax[1, 2].plot(hist_r, c='red')
ax[1, 2].plot(hist_g, c='green')
ax[1, 2].plot(hist_b, c='blue')

lut_prog = np.zeros((L), dtype=int)
lut_prog[50:200] = L-1
ax[2,0].plot(lut_base, lut_prog)
prog_image = lut_prog[raw_image]
ax[2,1].imshow(prog_image, cmap='binary_r')

hist_r, hist_g, hist_b = hist(prog_image)
ax[2, 2].plot(hist_r, c='red')
ax[2, 2].plot(hist_g, c='green')
ax[2, 2].plot(hist_b, c='blue')

sin_base = np.linspace(0, 2*np.pi, L)
lut_sin = np.sin(sin_base)
lut_sin = lut_sin+1
lut_sin = (lut_sin*((L-1)/2)).astype(int)
ax[3,0].plot(lut_base, lut_sin)
sin_image = lut_sin[raw_image]
ax[3,1].imshow(sin_image, cmap='binary_r')

hist_r, hist_g, hist_b = hist(sin_image)
ax[3, 2].plot(hist_r, c='red')
ax[3, 2].plot(hist_g, c='green')
ax[3, 2].plot(hist_b, c='blue')

gamma = 0.3
lut_gamma = np.linspace(255,0,L).astype(int)
lut_gamma = (((lut_base/(L-1)) ** (1/gamma)) * (L-1)).astype(int)
ax[4,0].plot(lut_base, lut_gamma)
gamma_image = lut_gamma[raw_image]
ax[4,1].imshow(gamma_image, cmap='binary_r')

hist_r, hist_g, hist_b = hist(gamma_image)
ax[4, 2].plot(hist_r, c='red')
ax[4, 2].plot(hist_g, c='green')
ax[4, 2].plot(hist_b, c='blue')

gamma = 3
lut_gamma = np.linspace(255,0,L).astype(int)
lut_gamma = (((lut_base/(L-1)) ** (1/gamma)) * (L-1)).astype(int)
ax[5,0].plot(lut_base, lut_gamma)
gamma_image = lut_gamma[raw_image]
ax[5,1].imshow(gamma_image, cmap='binary_r')

hist_r, hist_g, hist_b = hist(gamma_image)
ax[5, 2].plot(hist_r, c='red')
ax[5, 2].plot(hist_g, c='green')
ax[5, 2].plot(hist_b, c='blue')


plt.savefig("lab3.png")
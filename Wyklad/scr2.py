from skimage.data import chelsea
import numpy as np
import matplotlib.pyplot as plt

# parametry obrazu
D = 8
L = np.power(2, D).astype(int)

# Helpens
def hist(image):
    hist = np.unique(image, return_counts=True)
    vhist = np.zeros((L))
    vhist[hist[0]] = hist[1]
    vhist /= np.sum(vhist)
    vdist = np.cumsum(vhist)
    return vdist

def monochrome(raw_image, monohrome_transform=[1,1,1]):
    monochrome_transform = np.array([1,1,1]) #jakoś innaczej na wykłądzie
    monochrome_transform = monochrome_transform / np.sum(monochrome_transform)

    monochrome_transform = monochrome_transform / np.sum(monochrome_transform)
    mono_image = raw_image * monochrome_transform[None, None]
    mono_image = np.sum(mono_image, axis=-1).astype(np.uint8)
    return mono_image

fig, ax = plt.subplots(4, 2, figsize=(12,12))

# wczytujemy obraz
raw_image = monochrome(chelsea())

ax[0,0].imshow(raw_image, cmap='binary_r')

lut_base = np.arange(0,L)
#lut_negation = np.linspace(1.,0.,L)
lut_negation = np.linspace(255,0,L).astype(np.uint8)
#print(lut_base)
#print(lut_negation)

ax[0,1].scatter(lut_base[::8], lut_negation[::8], c='black', marker='x')

#print(raw_image.shape)
#print(lut_negation.shape)

negated_image = lut_negation[raw_image] # NAJWAŻNIEJSZE
ax[1,0].imshow(negated_image, cmap='binary_r')

anegated_image = lut_negation[negated_image]
ax[2,0].imshow(anegated_image, cmap='binary_r')

gamma = 0.5
lut_gamma = np.linspace(255,0,L).astype(np.uint8)
lut_gamma = (((lut_base/(L-1)) ** (1/gamma)) * (L-1)).astype(np.uint8)
ax[1,1].scatter(lut_base[::8], lut_gamma[::8], c='black', marker='x')

gamma_image = lut_gamma[raw_image]
ax[2,1].imshow(gamma_image, cmap='binary_r')

plt.savefig('foo2.png')
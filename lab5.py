import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage

fig, ax = plt.subplots(3, 4, figsize = (12, 12))

n = 100
x = np.linspace(0,11*np.pi,n)
sin = np.sin(x)
img = sin[:, np.newaxis]*sin[np.newaxis, :]

#normalizacja

min_img = np.min(img)
max_img = np.max(img)
img_norm = (img - min_img)
img_norm /= np.max(img_norm)
min_img_norm = round(np.min(img_norm), 3)
max_img_norm = round(np.max(img_norm), 3)

wsp = pow(2, 8)-1
m_8bit = np.rint(img_norm*wsp)
min_m_8bit = np.min(m_8bit)
max_m_8bit = np.max(m_8bit)

ax[0, 0].imshow(m_8bit, cmap = 'binary_r')

ft = np.fft.fft2(m_8bit)
ft = np.fft.fftshift(ft)

abs = np.abs(ft)

ax[0, 1].imshow(abs, cmap = 'binary_r')

log = np.log(abs)

ax[0, 2].imshow(log, cmap = 'binary_r')

#zad 2

lin = np.linspace(0, 11*np.pi, 100)
x, y = np.meshgrid(lin, lin)

ampl = [2, 4, 7, 9, 10]
angl = [0.1*np.pi, 0.15*np.pi, 1.17*np.pi, 2.36*np.pi, 4*np.pi]
wave = [0.5, 1, 1.2, 4, 8]
out_matrix = np.zeros((100, 100))

for i in range(5):
    out_matrix += ampl[i]*np.sin(2*np.pi*(x*np.cos(angl[i])+y*np.sin(angl[i]))*(1/wave[i]))

ax[1, 0].imshow(out_matrix, cmap = 'binary_r')

ft1 = np.fft.fft2(out_matrix)
ft1 = np.fft.fftshift(ft1)

abs1 = np.abs(ft1)

ax[1, 1].imshow(abs1, cmap = 'binary_r')

log1 = np.log(abs1)

ax[1, 2].imshow(log1, cmap = 'binary_r')

camera = skimage.data.camera()
ax[2, 0].imshow(camera, cmap = 'binary_r')

ft2 = np.fft.fft2(camera)
ft2 = np.fft.fftshift(ft2)

abs2 = np.abs(ft2)

ax[2, 1].imshow(abs2, cmap = 'binary_r')

log2 = np.log(abs2)

ax[2, 2].imshow(log2, cmap = 'binary_r')

#zad 3
def zad3(ft):
    red_ift = np.fft.ifftshift(ft.real)
    red_ift = np.fft.ifft2(red_ift)
    red_iabs = np.abs(red_ift)

    min_red_iabs = np.min(red_iabs)
    max_red_iabs = np.max(red_iabs)
    red_iabs_norm = (red_iabs - min_red_iabs)
    red_iabs_norm /= np.max(red_iabs_norm)

    ft_temp = np.copy(ft)
    ft_temp.real = 0
    green_ift = np.fft.ifftshift(ft_temp)
    green_ift = np.fft.ifft2(green_ift)
    green_iabs = np.abs(green_ift)

    min_green_iabs = np.min(green_iabs)
    max_green_iabs = np.max(green_iabs)
    green_iabs_norm = (green_iabs - min_green_iabs)
    green_iabs_norm /= np.max(green_iabs_norm)

    blue_ift = np.fft.ifftshift(ft)
    blue_ift = np.fft.ifft2(blue_ift)
    blue_iabs = np.abs(blue_ift)

    min_blue_iabs = np.min(blue_iabs)
    max_blue_iabs = np.max(blue_iabs)
    blue_iabs_norm = (blue_iabs - min_blue_iabs)
    blue_iabs_norm /= np.max(blue_iabs_norm)

    result = np.empty((ft.shape[0], ft.shape[1], 3)).astype(float)
    result[:,:,0] = red_iabs_norm
    result[:,:,1] = green_iabs_norm
    result[:,:,2] = blue_iabs_norm
    return result

    #out_img = np.concatenate((red_iabs_norm, green_iabs_norm, blue_iabs_norm), axis=1)
out_img = zad3(ft)

ax[0, 3].imshow(out_img)

out_img = zad3(ft1)

ax[1, 3].imshow(out_img)

out_img = zad3(ft2)

ax[2, 3].imshow(out_img)


from skimage.data import chelsea
raw_image = chelsea()
print(raw_image.shape)

plt.savefig("lab5.png")
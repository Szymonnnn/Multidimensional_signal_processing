import numpy as np
import matplotlib.pyplot as plt
import scipy

img = plt.imread("vessel.jpeg")
mono = np.mean(img, axis=2)

S1 = np.array([[-1,  0,  1],
               [-2,  0,  2],
               [-1,  0,  1]])

S2 = np.array([[ 0,  1,  2],
               [-1,  0,  1],
               [-2, -1,  0]])

S3 = np.array([[ 1,  2,  1],
               [ 0,  0,  0],
               [-1, -2, -1]])

S4 = np.array([[ 2,  1,  0],
               [ 1,  0, -1],
               [ 0, -1, -2]])

S1_img = scipy.ndimage.convolve(mono, S1)
S2_img = scipy.ndimage.convolve(mono, S2)
S3_img = scipy.ndimage.convolve(mono, S3)
S4_img = scipy.ndimage.convolve(mono, S4)

fig, ax = plt.subplots(3, 4, figsize = (12, 12))

ax[0,0].imshow(S1_img, cmap='binary_r')
ax[0,1].imshow(S2_img, cmap='binary_r')
ax[0,2].imshow(S3_img, cmap='binary_r')
ax[0,3].imshow(S4_img, cmap='binary_r')

def korelacja(mono_img, jadro):
    
    x_size = mono_img.shape[1]
    y_size = mono_img.shape[0]
    korel_img = np.zeros((y_size-2, x_size-2))
    for i in range(x_size-2):
        for j in range(y_size-2):
            korel_img[j][i] = (mono_img[j][i]*jadro[0][0]+
                               mono_img[j][i+1]*jadro[0][1]+
                               mono_img[j][i+2]*jadro[0][2]+
                               mono_img[j+1][i]*jadro[1][0]+
                               mono_img[j+1][i+1]*jadro[1][1]+
                               mono_img[j+1][i+2]*jadro[1][2]+
                               mono_img[j+2][i]*jadro[2][0]+
                               mono_img[j+2][i+1]*jadro[2][1]+
                               mono_img[j+2][i+2]*jadro[2][2])
    return korel_img

def konwolucja_ladnie(mono_img, jadro):
    if(jadro.shape[1]>mono_img.shape[1] and jadro.shape[0]>mono_img.shape[0]):
        jadro_temp = jadro
        jadro = mono_img
        mono_img = jadro_temp

    jadro = np.flip(jadro)

    x_jadro_size = jadro.shape[1]
    y_jadro_size = jadro.shape[0]
    
    x_size = mono_img.shape[1]
    y_size = mono_img.shape[0]
    korel_img = np.zeros((y_size-y_jadro_size+1, x_size-x_jadro_size+1))
    for i in range(x_size-x_jadro_size+1):
        for j in range(y_size-y_jadro_size+1):
            pixel = 0
            for i_ in range(x_jadro_size):
                for j_ in range(y_jadro_size):
                    pixel += (jadro[j_][i_] * mono_img[j+j_][i+i_])
            korel_img[j][i]=pixel
    return korel_img

S1_korel_img = korelacja(mono, S1)
S2_korel_img = korelacja(mono, S2)
S3_korel_img = korelacja(mono, S3)
S4_korel_img = korelacja(mono, S4)

ax[1,0].imshow(S1_korel_img, cmap='binary_r')
ax[1,1].imshow(S2_korel_img, cmap='binary_r')
ax[1,2].imshow(S3_korel_img, cmap='binary_r')
ax[1,3].imshow(S4_korel_img, cmap='binary_r')

S1_konw_img = konwolucja_ladnie(S1, mono)
S2_konw_img = konwolucja_ladnie(mono, S2)
S3_konw_img = konwolucja_ladnie(mono, S3)
S4_konw_img = konwolucja_ladnie(mono, S4)

ax[2,0].imshow(S1_konw_img, cmap='binary_r')
ax[2,1].imshow(S2_konw_img, cmap='binary_r')
ax[2,2].imshow(S3_konw_img, cmap='binary_r')
ax[2,3].imshow(S4_konw_img, cmap='binary_r')


plt.savefig("lab4.png")
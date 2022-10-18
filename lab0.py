import numpy as np

macierz = np.zeros([30, 30], dtype= int)
macierz[10:20, 10:20] = 1
macierz[15:25, 15:25] = 2
print(macierz)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,2, figsize=(7,7))

ax[0,0].imshow(macierz)
ax[0,0].set_title("obraz monochromatyczny")
ax[0,1].imshow(macierz, cmap = 'binary')
ax[0,1].set_title("obraz monochromatyczny")

color = np.zeros([30, 30, 3], dtype= float)

color[15:25, 5:15, 0] = 1
color[10:20, 10:20, 1] = 1
color[5:15, 15:25, 2] = 1

ax[1,0].imshow(color)
ax[1,0].set_title("obraz barwny")
ax[1,1].imshow(1-color)
ax[1,1].set_title("negatyw")

plt.savefig("lab0.png")
plt.show()

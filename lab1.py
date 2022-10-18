import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3,3, figsize=(10, 10))

#funkcja sinus
x = np.linspace(0, 4*np.pi, 40)
y = np.sin(x)
ax[0, 0].plot(x, y)
ax[0, 0].set_title("funkcja sinus")

#dwuwymiarowe złożenie spróbkowanej funkcji
macierz = y[:, np.newaxis]*y[np.newaxis, :]
min_macierz = round(np.min(macierz), 3)
max_macierz = round(np.max(macierz), 3)
ax[0, 1].imshow(macierz, cmap = 'binary')
ax[0, 1].set_title("min: " + str(min_macierz) + ", max: " + str(max_macierz))

#normalizacja
macierz_norm = (macierz - min_macierz)
macierz_norm /= np.max(macierz_norm)
min_macierz_norm = round(np.min(macierz_norm), 3)
max_macierz_norm = round(np.max(macierz_norm), 3)
ax[0, 2].imshow(macierz_norm, cmap = 'binary')
ax[0, 2].set_title("min: " + str(min_macierz_norm) + ", max: " + str(max_macierz_norm))


#plt.savefig("lab1.png")
plt.show()

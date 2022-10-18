import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3,3, figsize=(10, 10))

x = np.linspace(0, 4*np.pi, 40)
y = np.sin(x)

ax[0, 0].plot(x, y)
ax[0, 0].set_title("funkcja sinus")

macierz = y[:, np.newaxis]*y[np.newaxis, :]
min_macierz = round(np.min(macierz), 3)
max_macierz = round(np.max(macierz), 3)
print(min_macierz, max_macierz)

ax[0, 1].imshow(macierz, cmap = 'binary')
ax[0, 1].set_title("min: " + str(min_macierz) + ", max: " + str(max_macierz))


#plt.savefig("lab1.png")
plt.show()

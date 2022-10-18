import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3,3, figsize=(10, 10))
fig.subplots_adjust(hspace=.4)

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

#2-bit
wsp = 2*2-1
m_2bit = np.rint(macierz_norm*wsp)
min_m_2bit = np.min(m_2bit)
max_m_2bit = np.max(m_2bit)
ax[1, 0].imshow(m_2bit, cmap = 'binary')
ax[1, 0].set_title("min: " + str(min_m_2bit) + ", max: " + str(max_m_2bit))

#4-bit
wsp = 2*2*2*2-1
m_4bit = np.rint(macierz_norm*wsp)
min_m_4bit = np.min(m_4bit)
max_m_4bit = np.max(m_4bit)
ax[1, 1].imshow(m_4bit, cmap = 'binary')
ax[1, 1].set_title("min: " + str(min_m_4bit) + ", max: " + str(max_m_4bit))

#8-bit
wsp = 2*2*2*2*2*2*2*2-1
m_8bit = np.rint(macierz_norm*wsp)
min_m_8bit = np.min(m_8bit)
max_m_8bit = np.max(m_8bit)
ax[1, 2].imshow(m_8bit, cmap = 'binary')
ax[1, 2].set_title("min: " + str(min_m_8bit) + ", max: " + str(max_m_8bit))

m_8bit = (m_8bit - np.min(m_8bit))
m_8bit /= np.max(m_8bit)

#random normal
m_noise = np.random.normal(m_8bit)
ax[2, 0].imshow(m_noise, cmap = 'binary')
ax[2, 0].set_title("noised")

#adding noised images
m_50_noise = m_8bit
for i in range(50):
    m_50_noise = m_50_noise + np.random.normal(m_8bit)
#normalizacja
m_50_noise = (m_50_noise - np.min(m_50_noise))
m_50_noise /= np.max(m_50_noise)

ax[2, 1].imshow(m_50_noise, cmap = 'binary')
ax[2, 1].set_title("n=50")

#1000
m_1000_noise = m_8bit
for i in range(1000):
    m_1000_noise = m_1000_noise + np.random.normal(m_8bit)
m_1000_noise = (m_1000_noise - np.min(m_1000_noise))
m_1000_noise /= np.max(m_1000_noise)

ax[2, 2].imshow(m_1000_noise, cmap = 'binary')
ax[2, 2].set_title("n=1000")

plt.savefig("lab1.png")
plt.show()

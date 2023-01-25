import numpy as np
import matplotlib.pyplot as plt
import skimage
import random

img = np.zeros((100, 100, 3)).astype(int)
ground_truth = np.zeros((100, 100)).astype(int)

for i in range(3):
    promien = random.randint(10, 40)
    x = random.randint(promien, 100-promien)
    y = random.randint(promien, 100-promien)
    rr, cc = skimage.draw.disk((x, y), promien, shape=img.shape)
    wartosc_pikseli = random.randint(100, 255)
    img[rr,cc,random.randint(0,2)]+=wartosc_pikseli
    ground_truth[rr,cc]+=wartosc_pikseli

ground_truth = skimage.measure.label(ground_truth)

fig, ax = plt.subplots(2, 3, figsize = (12, 10))

mean, standard_deviation, samples = 0, 16, img.shape

img = img + np.random.normal(loc=mean,scale=standard_deviation, size=samples).astype(int)
img = np.clip(img, a_max=255, a_min=0)


ax[0,0].imshow(img)
ax[0,0].set_title("image")

ax[0,1].imshow(ground_truth)
ax[0,1].set_title("ground truth")


x = np.reshape(img, (100*100, 3))

xx, yy = np.meshgrid(np.arange(100), np.arange(100))

xx = xx.flatten()
yy = yy.flatten()


x= np.concatenate((x, xx[:,np.newaxis], yy[:,np.newaxis]), axis=1).astype(float)
y= ground_truth.flatten()

x-=np.mean(x)
x/=np.std(x)

print(x.shape, y.shape)
print(x[0], y[0])

from sklearn.cluster import KMeans, MiniBatchKMeans, Birch, DBSCAN
from sklearn.metrics import adjusted_rand_score

k_means_labels = KMeans().fit_predict(x)
k_means_score = adjusted_rand_score(k_means_labels, y)
k_means_labels = np.reshape(k_means_labels, (100, 100))
ax[0,2].imshow(k_means_labels)
ax[0,2].set_title("KMeans "+ str(round(k_means_score, 3)))

MiniBatchKMeans_labels = MiniBatchKMeans().fit_predict(x)
MiniBatchKMeans_score = adjusted_rand_score(MiniBatchKMeans_labels, y)
MiniBatchKMeans_labels = np.reshape(MiniBatchKMeans_labels, (100, 100))
ax[1,0].imshow(MiniBatchKMeans_labels)
ax[1,0].set_title("MiniBatchKM "+ str(round(MiniBatchKMeans_score, 3)))

Birch_labels = Birch().fit_predict(x)
Birch_score = adjusted_rand_score(Birch_labels, y)
Birch_labels = np.reshape(Birch_labels, (100, 100))
ax[1,1].imshow(Birch_labels)
ax[1,1].set_title("Birch "+ str(round(Birch_score, 3)))

DBSCAN_labels = DBSCAN().fit_predict(x)
DBSCAN_score = adjusted_rand_score(DBSCAN_labels, y)
DBSCAN_labels = np.reshape(DBSCAN_labels, (100, 100))
ax[1,2].imshow(DBSCAN_labels)
ax[1,2].set_title("DBSCAN "+ str(round(DBSCAN_score, 3)))

plt.savefig("lab11.png")
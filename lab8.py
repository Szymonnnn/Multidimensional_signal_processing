from sklearn.datasets import make_classification
from sklearn import datasets
X, y = datasets.make_classification(
    weights=[0.8,0.2],
    n_samples=100,
    n_features=2,
    n_informative=1,
    n_repeated=0,
    n_redundant=0,
    flip_y=.05,
    random_state=1234,
    n_clusters_per_class=1
)
print(X.shape)
print(y)

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.decomposition import PCA

img = loadmat('SalinasA_corrected.mat')['salinasA_corrected']

import time

print(img.shape)
fig, ax = plt.subplots(2, 3, figsize = (12, 12))

ax[0,0].imshow(img[:,:,9], cmap='binary_r')
ax[0,1].imshow(img[:,:,99], cmap='binary_r')
ax[0,2].imshow(img[:,:,199], cmap='binary_r')

x1 = img[10,10,:]
x2 = img[40,40,:]
x3 = img[80,80,:]

ax[1, 0].plot(x1)
ax[1, 1].plot(x2)
ax[1, 2].plot(x3)

plt.savefig("lab8.png")

fig, ax = plt.subplots(1, 2, figsize = (12, 12))

r = img[:,:,4]
g = img[:,:,12]
b = img[:,:,26]

min_r = np.min(r)
max_r = np.max(r)
r_norm = (r - min_r)
r_norm = r_norm / np.max(r_norm)

min_g = np.min(g)
max_g = np.max(g)
g_norm = (g - min_g)
g_norm = g_norm / np.max(g_norm)

min_b = np.min(b)
max_b = np.max(b)
b_norm = (b - min_b)
b_norm = b_norm / np.max(b_norm)

result = np.empty((r.shape[0], r.shape[1], 3)).astype(float)
result[:,:,0] = r_norm
result[:,:,1] = g_norm
result[:,:,2] = b_norm

ax[0].imshow(result)

img_res = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
print(img_res.shape)
pca = PCA(n_components=3)
X =pca.fit_transform(img_res)
print(X.shape)
X = X.reshape(img.shape[0], img.shape[1], 3)
print(X.shape)

r = X[:,:,0]
g = X[:,:,1]
b = X[:,:,2]

min_r = np.min(r)
max_r = np.max(r)
r_norm = (r - min_r)
r_norm = r_norm / np.max(r_norm)

min_g = np.min(g)
max_g = np.max(g)
g_norm = (g - min_g)
g_norm = g_norm / np.max(g_norm)

min_b = np.min(b)
max_b = np.max(b)
b_norm = (b - min_b)
b_norm = b_norm / np.max(b_norm)

result2 = np.empty((r.shape[0], r.shape[1], 3)).astype(float)
result2[:,:,0] = r_norm
result2[:,:,1] = g_norm
result2[:,:,2] = b_norm

ax[1].imshow(result2)

plt.savefig("lab8_2.png")

label = loadmat('SalinasA_gt.mat')['salinasA_gt']
print(label)

X = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
y = label.reshape(img.shape[0]*img.shape[1])
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.30,
    random_state=1234
)

rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=1234)
scores = []

clf = GaussianNB()
for train_index, test_index in rkf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    scores.append(accuracy_score(y_test, predict))

mean_score = np.mean(scores)
std_score = np.std(scores)
print("All: %.3f (%.3f)" % (mean_score, std_score))

X = result.reshape(result.shape[0]*result.shape[1], result.shape[2])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.30,
    random_state=1234
)

rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=1234)
scores = []

clf = GaussianNB()
for train_index, test_index in rkf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    scores.append(accuracy_score(y_test, predict))

mean_score = np.mean(scores)
std_score = np.std(scores)
print("RGB: %.3f (%.3f)" % (mean_score, std_score))

X = result2.reshape(result2.shape[0]*result2.shape[1], result2.shape[2])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.30,
    random_state=1234
)

rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=1234)
scores = []

clf = GaussianNB()
for train_index, test_index in rkf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    scores.append(accuracy_score(y_test, predict))

mean_score = np.mean(scores)
std_score = np.std(scores)
print("PCA: %.3f (%.3f)" % (mean_score, std_score))
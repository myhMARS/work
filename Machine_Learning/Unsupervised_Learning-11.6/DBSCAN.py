import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

X = np.loadtxt("kmeans_data.csv", delimiter=',')
outliers = np.array([
    [8, 8],
    [-8, -8],
    [8, -8],
    [-8, 8]
])

dbscan = DBSCAN(eps=1.5, min_samples=3)
labels = dbscan.fit_predict(X)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]
    plt.plot(X[labels == k, 0], X[labels == k, 1], 'o', markerfacecolor=tuple(col))
plt.title("DBSCAN Clustering")
plt.show()

from kmeans import X, random_init, show_cluster, Kmeans
import numpy as np

K = 4
outliers = np.array([
    [8, 8],
    [-8, -8],
    [8, -8],
    [-8, 8]
])
X = np.vstack((X, outliers))
init_centroids = random_init(X, K)
cent, cluster,

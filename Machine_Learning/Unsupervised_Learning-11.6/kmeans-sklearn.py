from sklearn.cluster import KMeans
from kmeans import X, show_cluster

K = 4
kmeans = KMeans(n_clusters=K)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
show_cluster(X,labels,centroids=centers)

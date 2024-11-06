import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt('kmeans_data.csv', delimiter=',')
print('数据集大小：', len(X))

print(X[0:3])

# 绘图函数
def show_cluster(X, cluster, centroids=None):  
    # X：数据
    # cluster：每个样本所属聚类
    # centroids：聚类中心点的坐标
    K = len(np.unique(cluster))
    colors = ['blue', 'red', 'green', 'purple']
    markers = ['o', '^', 's', 'd']
    
    for i in range(K):
        # 画第 i 个簇的
        plt.scatter(X[cluster == i, 0], X[cluster == i, 1], color=colors[i], marker=markers[i])
    
    # 画每个簇的质心
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], color=colors[0], marker='+', s=150) 
     
    plt.show()

show_cluster(X, np.zeros(len(X), dtype=int))

def random_init(X, K):
    np.random.seed(0)
    # 于从给定的一维数组（或整数序列）中随机抽取元素，replace=False 表示抽取是无放回的
    idx = np.random.choice(np.arange(len(X)), size=K, replace=False)
    return X[idx]


def Kmeans(X, K, init_centroids):
    centroids = init_centroids
    cluster = np.zeros(len(X), dtype=int)
    changed = True    
    itr = 0 # 开始迭代
    
    while changed:
        changed = False
        for i, data in enumerate(X):
            dis = np.sum((centroids - data) ** 2, axis=-1)
            k = np.argmin(dis)
            if cluster[i] != k:
                cluster[i] = k
                changed = True
        show_cluster(X, cluster, centroids)
        for i in range(K):
            centroids[i] = np.mean(X[cluster == i], axis=0)
        itr += 1

    return centroids, cluster


K = 4
init_centroids = random_init(X, K)
cent, cluster = Kmeans(X, K, init_centroids)

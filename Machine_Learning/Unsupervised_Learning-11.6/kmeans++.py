import timeit

import numpy as np
import matplotlib.pyplot as plt
from kmeans import random_init

# 数据加载
X = np.loadtxt('kmeans_data.csv', delimiter=',')
print('数据集大小：', len(X))
print(X[0:3])


# 绘图函数
def show_cluster(X, cluster, centroids=None):
    K = len(np.unique(cluster))
    colors = plt.cm.tab10.colors  # 动态生成颜色
    markers = ['o', '^', 's', 'd', 'p', 'x', '+', '*', 'h', 'v']

    for i in range(K):
        plt.scatter(X[cluster == i, 0], X[cluster == i, 1], color=colors[i % len(colors)],
                    marker=markers[i % len(markers)], label=f'Cluster {i}')

    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100, label='Centroids')

    plt.legend()
    plt.show()


# KMeans++ 初始化质心
def kmeans_plus_plus_init(X, K):
    np.random.seed(42)  # 固定随机种子
    n_samples = X.shape[0]
    centroids = []

    # 随机选择第一个质心
    first_centroid_idx = np.random.choice(n_samples)
    centroids.append(X[first_centroid_idx])

    # 迭代选择其余的质心
    for _ in range(1, K):
        # 计算每个点到最近质心的距离平方
        dist_sq = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
        # 根据距离平方的概率分布选择下一个质心
        probabilities = dist_sq / np.sum(dist_sq)
        next_centroid_idx = np.random.choice(n_samples, p=probabilities)
        centroids.append(X[next_centroid_idx])

    return np.array(centroids)


# K-Means 主函数
def Kmeans(X, K, init_centroids):
    centroids = init_centroids.copy()
    cluster = np.zeros(len(X), dtype=int)
    changed = True
    itr = 0
    loss_history = []

    while changed:
        changed = False
        loss = 0

        # 分配样本到最近质心
        for i, data in enumerate(X):
            dis = np.sum((centroids - data) ** 2, axis=-1)
            k = np.argmin(dis)
            if cluster[i] != k:
                cluster[i] = k
                changed = True
            loss += 0.5 * np.sum((data - centroids[k]) ** 2)

        loss_history.append(loss)
        # show_cluster(X, cluster, centroids)

        # 更新质心
        for i in range(K):
            if np.any(cluster == i):
                centroids[i] = np.mean(X[cluster == i], axis=0)
            else:
                centroids[i] = random_init(X, 1)  # 重新随机初始化空簇

        itr += 1
        print(f"第{itr}次迭代:loss={loss}")

    return centroids, cluster, loss_history


if __name__ == "__main__":
    K = 4
    init_centroids = kmeans_plus_plus_init(X, K)  # 使用 KMeans++ 初始化
    cent, cluster, loss_history = Kmeans(X, K, init_centroids)

    # 绘制损失下降曲线
    x = range(len(loss_history))
    y = loss_history
    plt.plot(x, y, label='Loss', color='blue', linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

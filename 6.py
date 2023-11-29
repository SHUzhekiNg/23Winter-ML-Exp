from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# 加载Iris数据集
iris = datasets.load_iris()
X = iris.data

# 创建K均值模型，假设有3个簇（因为Iris数据集有3个类别）
kmeans = KMeans(n_clusters=2, random_state=42)

# 拟合模型
kmeans.fit(X)

# 获取簇中心
centroids = kmeans.cluster_centers_

# 获取每个样本所属的簇
labels = kmeans.labels_

# 绘制聚类结果
plt.figure(figsize=(8, 6))

# 绘制数据点
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.Paired, edgecolors='k', s=100)
# 绘制簇中心
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='red', label='Centroids')

plt.title('K-Means Clustering of Iris Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

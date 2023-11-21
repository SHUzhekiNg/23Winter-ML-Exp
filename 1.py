import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 下载iris数据集
iris = load_iris()
X = iris.data
y = iris.target


print(iris)
# 选择两个维度进行可视化，例如第一个和第二个特征
feature1 = 0
feature2 = 1

# 创建散点图
plt.figure(figsize=(8, 6))
for i in range(len(iris.target_names)):
    plt.scatter(X[y == i, feature1], X[y == i, feature2], label=iris.target_names[i])

plt.title('Iris Dataset - Two Dimensional Visualization')
plt.xlabel(iris.feature_names[feature1])
plt.ylabel(iris.feature_names[feature2])
plt.legend()
plt.show()

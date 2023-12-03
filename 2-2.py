import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score

# 加载Iris数据集
iris = load_iris()
X = iris.data
y = iris.target

# 只选择两个类别（setosa和versicolor）X只取前两列
X = X[y != 2][:,:2]
y = y[y != 2]

# 将类别转换为0和1
y[y == 0] = -1  # 将setosa类别设为-1
y[y == 1] = 1   # 将versicolor类别设为1

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42) # 

# 使用最小二乘法拟合线性回归模型
model = Lasso()
model.fit(X_train, y_train)

# 预测
y_pred = np.sign(model.predict(X_test))  # 使用符号函数将预测结果转换为0和1

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 定义决策边界函数
def decision_boundary(x):
    return (-model.coef_[0] * x - model.intercept_) / model.coef_[1]

# 绘制决策边界
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, label='Training Set', edgecolors='k', s=100)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired, label='Test Set', marker='x', edgecolors='k', s=100)

# 生成用于绘制决策边界的点
plot_x = np.array([X[:, 0].min() - 1, X[:, 0].max() + 1])
plot_y = decision_boundary(plot_x)

# 绘制决策边界
plt.plot(plot_x, plot_y, linestyle='--', color='black', label='Decision Boundary')

plt.title('Decision Boundary and Data Points')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import time


# 加载Iris数据集
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

bc = datasets.load_breast_cancer() 
X = bc.data
y = bc.target
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建朴素贝叶斯分类器
nb_classifier = GaussianNB()

# 拟合模型
st = time.time()
nb_classifier.fit(X_train, y_train)

# 预测
y_pred = nb_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

print(f"Gaussian Naive Bayes use {time.time()-st} s")

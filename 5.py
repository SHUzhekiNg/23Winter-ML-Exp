import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载 Iris 数据集
iris = load_breast_cancer()
X, y = iris.data, iris.target

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 的 Tensor
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义全连接层的神经网络
class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc = nn.Linear(30, 2)  # 输入特征为4，输出类别为3

    def forward(self, x):
        x = self.fc(x)
        return x

# 初始化网络、损失函数和优化器
fc_model = FCNet()
criterion_fc = nn.CrossEntropyLoss()
optimizer_fc = optim.SGD(fc_model.parameters(), lr=0.01)

# 训练全连接层的神经网络
epochs = 500
for epoch in range(epochs):
    for inputs, labels in train_loader:
        optimizer_fc.zero_grad()
        outputs = fc_model(inputs)
        loss = criterion_fc(outputs, labels)
        loss.backward()
        optimizer_fc.step()

# 在测试集上进行测试
with torch.no_grad():
    fc_model.eval()
    y_pred = torch.argmax(fc_model(X_test), axis=1)
    accuracy = torch.sum(y_pred == y_test).item() / len(y_test)
    print(f'FCNet Accuracy: {accuracy:.4f}')

# 定义卷积神经网络
# class CNNNet(nn.Module):
#     def __init__(self):
#         super(CNNNet, self).__init__()
#         self.conv1 = nn.Conv3d(1, 3, kernel_size=3)  # 调整卷积核大小为2x2
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(3, 2)  # 输入特征为3x2x2，输出类别为3

#     def forward(self, x):
#         x = x.view(-1, 1, 2, 2)  # 转换输入的形状
#         x = self.conv1(x)
#         x = torch.relu(x)
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x

# # 初始化网络、损失函数和优化器
# cnn_model = CNNNet()
# criterion_cnn = nn.CrossEntropyLoss()
# optimizer_cnn = optim.SGD(cnn_model.parameters(), lr=0.01)

# # 调整输入数据的形状
# X_train_cnn = X_train.view(-1, 1, 30, 1)
# X_test_cnn = X_test.view(-1, 1, 30, 1)

# # 创建数据加载器
# train_dataset_cnn = TensorDataset(X_train_cnn, y_train)
# train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=64, shuffle=True)

# # 训练卷积神经网络
# for epoch in range(epochs):
#     for inputs, labels in train_loader_cnn:
#         optimizer_cnn.zero_grad()
#         outputs = cnn_model(inputs)
#         loss = criterion_cnn(outputs, labels)
#         loss.backward()
#         optimizer_cnn.step()

# # 在测试集上进行测试
# with torch.no_grad():
#     cnn_model.eval()
#     y_pred = torch.argmax(cnn_model(X_test_cnn), axis=1)
#     accuracy = torch.sum(y_pred == y_test).item() / len(y_test)
#     print(f'CNNNet Accuracy: {accuracy:.4f}')

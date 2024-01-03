import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import metrics

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc = nn.Linear(4, 3)  # 输入特征为4，输出类别为3

    def forward(self, x):
        x = self.fc(x)
        return x
    
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=2)  # 调整卷积核大小为2x2
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3, 3)  # 输入特征为3x2x2，输出类别为3

    def forward(self, x):
        x = x.view(-1, 1, 2, 2)  # 转换输入的形状
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor, X_test_tensor = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train_tensor, y_test_tensor = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

fc_model = FCNet()
criterion_fc = nn.CrossEntropyLoss()
optimizer_fc = optim.SGD(fc_model.parameters(), lr=0.01)

epochs = 500
for epoch in range(epochs):
    for inputs, labels in train_loader:
        optimizer_fc.zero_grad()
        outputs = fc_model(inputs)
        loss = criterion_fc(outputs, labels)
        loss.backward()
        optimizer_fc.step()

with torch.no_grad():
    fc_model.eval()
    y_pred = torch.argmax(fc_model(X_test_tensor), axis=1)
    accuracy = torch.sum(y_pred == y_test_tensor).item() / len(y_test_tensor)
    print(f'FCNet Accuracy: {accuracy:.4f}')

cnn_model = CNNNet()
criterion_cnn = nn.CrossEntropyLoss()
optimizer_cnn = optim.SGD(cnn_model.parameters(), lr=0.01)

X_train_cnn = X_train_tensor.view(-1, 1, 4, 1)
X_test_cnn = X_test_tensor.view(-1, 1, 4, 1)

train_dataset_cnn = TensorDataset(X_train_cnn, y_train_tensor)
train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=64, shuffle=True)

for epoch in range(epochs):
    for inputs, labels in train_loader_cnn:
        optimizer_cnn.zero_grad()
        outputs = cnn_model(inputs)
        loss = criterion_cnn(outputs, labels)
        loss.backward()
        optimizer_cnn.step()

with torch.no_grad():
    cnn_model.eval()
    y_pred = torch.argmax(cnn_model(X_test_cnn), axis=1)
    accuracy = torch.sum(y_pred == y_test_tensor).item() / len(y_test_tensor)
    print(f'CNNNet Accuracy: {accuracy:.4f}')

clf = svm.SVC(gamma=0.001, C=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'SVM Accuracy: {accuracy:.4f}')
# print(metrics.classification_report(y_test, y_pred))

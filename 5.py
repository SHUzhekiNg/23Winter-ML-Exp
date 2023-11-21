import torch.nn as nn

class FullConnection(nn.Module):
    def __init__(self, in_size, out_size):
        super(FullConnection, self).__init__()
        self.fc = nn.Linear(in_size, out_size)

    def forward(self, x):
        x = self.fc(x)
        return x

# TODO:
class CNN(nn.Module):
    def __init__(self, in_size, out_size):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(in_size, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 28 * 28, out_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # 将卷积层输出展平成一维向量
        x = self.fc(x)
        return x
    

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# X为X_test.csv中的内容，读取X_test.csv
X = np.loadtxt('X_test.csv', delimiter=',',skiprows=1)
y = np.loadtxt('y_test.csv', delimiter=',',skiprows=1)

# 假设X是特征矩阵，形状为(12833, 10)，y是标签向量，形状为(12833,)
# 将数据转换为PyTorch的张量
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) 
y = torch.tensor(y, dtype=torch.long)

# 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 定义模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.fc1 = nn.Linear(64 * 3, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x = x.unsqueeze(1)  # 增加一个维度，变为(N, C, L)格式
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型
model = ConvNet()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    # 验证模型
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)
        _, predicted = torch.max(val_output, 1)
        correct = (predicted == y_val).sum().item()
        val_accuracy = correct / y_val.size(0)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}, Val Accuracy: {val_accuracy}')

# 测试模型
model.eval()
with torch.no_grad():
    test_output = model(X_test)
    _, predicted = torch.max(test_output, 1)
    correct = (predicted == y_test).sum().item()
    test_accuracy = correct / y_test.size(0)
print(f'Test Accuracy: {test_accuracy}')

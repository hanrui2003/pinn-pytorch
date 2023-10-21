import torch
import torch.nn as nn
import numpy as np


# 定义PINN模型
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # 全连接层
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, t):
        # 将 (x, t) 连接并传递给卷积层
        input_data = torch.cat((x, t), dim=1)
        out = input_data.unsqueeze(1)  # 添加通道维度
        out = self.pool(self.relu(self.conv1(out)))
        out = out.view(out.size(0), -1)  # 展平卷积输出

        # 传递给全连接层
        out = self.fc2(self.relu(self.fc1(out)))
        return out


# 定义波动方程函数
def wave_equation(x, t):
    return torch.sin(np.pi * x) * torch.cos(np.pi * t)


# 训练数据
x_data = torch.tensor(np.linspace(0, 1, 101).reshape(-1, 1), dtype=torch.float32)
t_data = torch.tensor(np.linspace(0, 1, 101).reshape(-1, 1), dtype=torch.float32)
u_data = wave_equation(x_data, t_data)

# 初始化PINN模型
model = PINN()

# 定义损失函数
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    u_pred = model(x_data, t_data)
    loss = criterion(u_pred, u_data)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 使用模型进行预测
x_test = torch.tensor(np.linspace(0, 1, 100).reshape(-1, 1), dtype=torch.float32)
t_test = torch.tensor(np.linspace(0, 1, 100).reshape(-1, 1), dtype=torch.float32)
u_pred = model(x_test, t_test)

# 打印部分预测结果
for i in range(10):
    print(f'x = {x_test[i].item():.4f}, t = {t_test[i].item():.4f}, u(x, t) = {u_pred[i].item():.4f}')

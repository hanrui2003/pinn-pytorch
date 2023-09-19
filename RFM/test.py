import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单的分支网络
class BranchNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BranchNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 定义整体多分支网络
class MultiBranchNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_branches):
        super(MultiBranchNetwork, self).__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList(
            [BranchNetwork(input_size, hidden_size, output_size) for _ in range(num_branches)])

    def forward(self, x):
        branch_outputs = []
        for i in range(self.num_branches):
            branch_output = self.branches[i](x[i])  # 每个分支处理不同区间的输入
            branch_outputs.append(branch_output)
        return branch_outputs


# 创建模拟数据，假设有3个不同区间的输入
input_data = [torch.randn(32, 10), torch.randn(32, 10), torch.randn(32, 10)]

# 创建多分支网络
input_size = 10
hidden_size = 20
output_size = 2
num_branches = len(input_data)
model = MultiBranchNetwork(input_size, hidden_size, output_size, num_branches)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 前向传播
branch_outputs = model(input_data)

# 计算损失
loss = sum([criterion(output, target) for output, target in zip(branch_outputs, target_data)])

# 反向传播
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("损失:", loss.item())

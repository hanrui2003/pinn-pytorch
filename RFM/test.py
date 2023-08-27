import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# 创建模型实例
model = MyModel()

# 查看模型参数的数据类型
print(model.fc.weight.dtype)  # 默认为 float32

# 将模型参数转换为 double 类型
model.double()

# 查看模型参数的数据类型
print(model.fc.weight.dtype)  # 现在为 float64

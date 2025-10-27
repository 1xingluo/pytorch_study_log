import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# ===== 数据准备 =====
x_data = torch.tensor([[1.0],[2.0],[3.0]])
x_data = x_data.to('cuda:0')
y_data = torch.tensor([[0.0],[0.0],[1.0]])  # ⚠️ 注意浮点数
y_data = y_data.to('cuda:0')
# ===== 定义模型 =====
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 1)
        self.actfn = nn.Sigmoid()
    def forward(self, x):
        return self.actfn(self.layer1(x))

# ===== 创建模型 =====
model = LogisticRegressionModel().to('cuda:0')

# ===== 损失函数 & 优化器 =====
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# ===== 训练循环 =====
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch}: loss={loss.item():.4f}')

# ===== 测试绘图 =====
x = np.linspace(0, 10, 200)
x_t = torch.tensor(x, dtype=torch.float32).view(-1, 1).to('cuda:0')

with torch.no_grad():  # 评估模式，不计算梯度
    y_t = model(x_t)
y = y_t.cpu().detach().numpy()

plt.plot(x, y, label='Predicted P(y=1)')
plt.plot([0,10],[0.5,0.5],c='r', linestyle='--', label='Decision boundary 0.5')
plt.xlabel('x')
plt.ylabel('Probability')
plt.grid(True)
plt.legend()
plt.show()

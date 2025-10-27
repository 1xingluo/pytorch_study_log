import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
myseed = 42069  # 随机种子
# NumPy 随机数
np.random.seed(myseed)
# PyTorch CPU 随机数
torch.manual_seed(myseed)
# PyTorch GPU 随机数
torch.cuda.manual_seed(myseed)
torch.cuda.manual_seed_all(myseed)  # 如果有多块 GPU
# 确保卷积操作可复现
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================
# 1. 数据读取与特征筛选
# =========================
data_tr = pd.read_csv('covid.train.csv', index_col=0)

# 第41列到最后列，排除前40个one-hot
corr_matrix = data_tr.iloc[:, 40:].corr()

# 锁定目标列名（最后一列）
target_name = data_tr.columns[-1]
target_corr = corr_matrix[target_name]

# 设置阈值筛选高相关特征
threshold = 0.7# 可以自己调节
# threshold 是相关性阈值
high_corr_features = target_corr[(target_corr > threshold) & (target_corr.index != target_name)].index.tolist()


# 合并前40列one-hot + 高相关特征
selected_cols = list(data_tr.columns[:40]) + high_corr_features
# 构造新数据集并保存
df_selected = data_tr[selected_cols + [target_name]]
df_selected.to_csv('covid_selected.csv', index=False)
print(f"原始特征数: {data_tr.shape[1]-1}, 筛选后特征数: {len(selected_cols)}")
print(list(selected_cols))
# =========================
# 2. PyTorch Dataset
# =========================
class MyData(Dataset):
    def __init__(self, filepath):
        df = pd.read_csv(filepath)
        arr = df.values.astype(np.float32)
        ts = torch.tensor(arr).to('cuda:0')
        self.X = ts[:, :-1]
        self.Y = ts[:, -1:]
        self.len = ts.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len

# 加载数据
Data = MyData('covid_selected.csv')
train_size = int(len(Data) * 0.7)
test_size = len(Data) - train_size
train_data, test_data = random_split(Data, [train_size, test_size])

train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=16)
test_loader = DataLoader(dataset=test_data, shuffle=False, batch_size=64)

# =========================
# 3. DNN 网络
# =========================
input_dim = len(selected_cols)

class DNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 70), nn.ReLU(),
            nn.Linear(70, 1),
        )

    def forward(self, x):
        return self.net(x)

model = DNN(input_dim).to('cuda:0')

# =========================
# 4. 损失函数 & 优化器
# =========================
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

# =========================
# 5. 训练模型
# =========================
epochs =600
losses = []

for epoch in range(epochs):
    epoch_loss = 0
    for x, y in train_loader:
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        epoch_loss += loss.item() * x.size(0)
    print('epoch=',epoch,'loss',epoch_loss / len(train_loader.dataset))    
# 可视化训练loss
plt.figure()
plt.plot(range(len(losses)), losses)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.show()

# =========================
# 6. 测试模型
# =========================
total_loss = 0
n = 0
with torch.no_grad():
    for x, y in test_loader:
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)

print("Test MSE:", total_loss / n)


test_df = pd.read_csv('covid.test.csv', index_col=0)

# 只保留训练特征中测试集存在的列
test_cols = [c for c in selected_cols if c in test_df.columns]

X_test = test_df[test_cols].values.astype(np.float32)
X_test = torch.tensor(X_test).to('cuda:0')

# 使用模型预测
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

y_pred = y_pred.cpu().numpy()

# 保存结果
output_df = pd.DataFrame(y_pred, columns=['tested_positive'])
output_df.to_csv('covid_test_pred.csv', index=False)
print("预测完成，已保存到 covid_test_pred.csv")

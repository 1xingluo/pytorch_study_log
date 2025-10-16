import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#引入csv数据
df=pd.read_csv('covid.train.csv',index_col=0)
df2=pd.read_csv('covid.test.csv',index_col=0)
#退化为ndarray
arr=df.values
arr1=df2.values
#类型变为float32
arr=arr.astype(np.float32)
arr1=arr1.astype(np.float32)
#数组变成张量
ts=torch.tensor(arr)
ts1=torch.tensor(arr1)
#转到gpu
ts=ts.to('cuda:0')
ts1=ts1.to('cuda:0')
print(ts.shape)
print(ts1.shape)

#划分训练集 测试集
train_size=(int)(len(ts)*0.7)
test_size=len(ts)-train_size
ts=ts[torch.randperm(ts.size(0)),:]
train_data=ts[:train_size,:]
test_data=ts[train_size:,:]
print('train_shape',train_data.shape)
print('test_shape',test_data.shape)
#搭建数据集网络
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(      #按顺序搭建各层
            nn.Linear(93,128),nn.ReLU(),#第一层：输入为x维度3*1，输出z1维度5*1
            nn.Linear(128,64),nn.ReLU(),#第二层：输入为z1维度5*1，输出z2维度5*1
            nn.Linear(64,32),nn.ReLU(),#第三层：输入为z2维度5*1，输出z3维度5*1
            nn.Linear(32,1),nn.ReLU(),#第一层：输入为z3维度5*1，输出y维度3*1
        )
    def forward(self,x):
        y_pred=self.net(x)
        return y_pred
model=DNN().to('cuda:0')
print(model)
learning_rate=0.01
criterion=nn.MSELoss(reduction='mean')
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
epochs=1000
losses=[]
X=train_data[:,:-1]
Y=train_data[:,-1:].reshape((-1,1))
for epoch in range(epochs):
    Pred=model(X)
    loss=criterion(Pred,Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
print(model.net)
X=ts1
# plt.figure()
# plt.plot(range(epochs),losses)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()
# X=test_data[:,:-1]
# Y=test_data[:,-1:]
# with torch.no_grad():
#     Pred=model(X)
#     loss=criterion(Pred,Y)
#     print(loss.item())
with torch.no_grad():
    Pred=model(X)
Pred_numpy = Pred.detach().cpu().numpy()
df_pred = pd.DataFrame(Pred_numpy, columns=['y_pred'])
df_pred.to_csv('predictions.csv', index=False)
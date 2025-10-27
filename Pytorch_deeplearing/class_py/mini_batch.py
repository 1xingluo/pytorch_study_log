import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
#制作数据集
class MyData(Dataset):
    def __init__(self,filepath):
        df=pd.read_csv(filepath,index_col=0)
        arr=df.values
        arr=arr.astype(np.float32)
        ts=torch.tensor(arr)
        ts=ts.to('cuda:0')
        self.X=ts[:,:-1]
        self.Y=ts[:,-1:]
        self.len=ts.shape[0]
    def __getitem__(self, index):
        return self.X[index],self.Y[index]
    def __len__(self):
        return self.len
#划分训练集和测试集：
Data=MyData('covid.train.csv')
train_size=int(len(Data)*0.7)
test_size=len(Data)-train_size
train_data,test_data=random_split(Data,[train_size,test_size])
#批量加载器：
train_loader=DataLoader(dataset=train_data,shuffle=True,batch_size=128)
test_loader=DataLoader(dataset=test_data,shuffle=False,batch_size=64)
#搭建神经网络
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
#损失函数，优化算法：
criterion=nn.MSELoss(reduction='mean')

learning_rate=0.01
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
#训练网络
epochs=500
losses=[]
for epoch in range(epochs):
    for(x,y) in train_loader:
        Pred=model(x)
        loss=criterion(Pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
# fig=plt.figure()
# plt.plot(range(len(losses)),losses)
# plt.show()

#测试网络：
total_loss = 0
n = 0
with torch.no_grad():
    for x, y in test_loader:
        Pred = model(x)
        loss = criterion(Pred, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
print("Test MSE:", total_loss / n)
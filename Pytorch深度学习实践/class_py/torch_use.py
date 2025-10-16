import torch
import torch.nn as nn
import matplotlib.pyplot as plt
#1.3 GPU 存储张量：
# ts1=torch.randn(3,4)
# print(ts1)
# ts2=ts1.to('cuda:0')
# print(ts2)

#3.DNN实现：例子是批量梯度下降（all）   还有随机梯度下降batch=1,小批量batch=N
#3.1生成数据集
x1=torch.rand(10000,1)
x2=torch.rand(10000,1)
x3=torch.rand(10000,1)
y1=((x1+x2+x3)<1).float()
y2=((1<(x1+x2+x3))&((x1+x2+x3)<2)).float()
y3=((x1+x2+x3)>2).float()
Data=torch.cat([x1,x2,x3,y1,y2,y3],axis=1)
Data=Data.to('cuda:0')
print('Data.data',Data.data)
print('Data.shape',Data.shape)

train_size=(int)(len(Data)*0.7)
test_size=len(Data)-train_size
Data=Data[torch.randperm(Data.size(0)),:]
train_data=Data[:train_size,:]
test_data=Data[train_size:,:]
print('train_shape',train_data.shape)
print('test_shape',test_data.shape)

#3.2搭建数据集网络
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(      #按顺序搭建各层
            nn.Linear(3,5),nn.ReLU(),#第一层：输入为x维度3*1，输出z1维度5*1
            nn.Linear(5,5),nn.ReLU(),#第二层：输入为z1维度5*1，输出z2维度5*1
            nn.Linear(5,5),nn.ReLU(),#第三层：输入为z2维度5*1，输出z3维度5*1
            nn.Linear(5,3),#第一层：输入为z3维度5*1，输出y维度3*1
        )
    def forward(self,x):
        y_pred=self.net(x)
        return y_pred
model=DNN().to('cuda:0')
print(model)
#3.3神经网络内部参数：
for name,param in model.named_parameters():
    print(f"参数:{name}\n形状:{param.shape}\n数值:{param}\n")

#3.4网络外部参数：超参数 激活函数，损失函数，学习率，优化算法
learning_rate=0.01
criterion=nn.MSELoss(reduction='mean')
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
#3.5训练网络
epochs=1000
losses=[]
X=train_data[:,:3]
Y=train_data[:,-3:]
for epoch in range(epochs):
    Pred=model(X)
    loss=criterion(Pred,Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
print(model.net)
#3.6测试网络：
X=test_data[:,:3]
Y=test_data[:,-3:]
# with torch.no_grad():
#     Pred=model(X)
#     Pred[:,torch.argmax(Pred,axis=1)]=1
#     Pred[Pred!=1]=0
#     corrent=torch.sum((Pred==Y).all(1))
#     total=Y.size(0)
#     print('测试正确率：',(100*corrent/total).item(),'%')

#3.7保存网络
torch.save(model,'model.pth')
#把模型赋值网络
new_model=torch.load('model.pth')
with torch.no_grad():
    Pred=new_model(X)
    Pred[:,torch.argmax(Pred,axis=1)]=1
    Pred[Pred!=1]=0
    corrent=torch.sum((Pred==Y).all(1))
    total=Y.size(0)
    print('测试正确率：',(100*corrent/total).item(),'%')

plt.figure()
plt.plot(range(epochs),losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

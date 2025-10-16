import torch
x_data=torch.tensor([[1.0],[2.0],[3.0]])
y_data=torch.tensor([[2.0],[4.0],[6.0]])
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()#初始化父类
        self.Linear=torch.nn.Linear(1,1)#定义网络模型，构建Linear Unit
    def forward(self,x):
        y_pred=self.Linear(x) #前向
        return y_pred
#创建一个模型：
Model=LinearModel()
#创建loss模型
criterion=torch.nn.MSELoss(size_average=False)
#优化器：
optimizer=torch.optim.SGD(Model.parameters(),lr=0.01,momentum=0.8)
for epoch in range(100):
    y_pred=Model(x_data)
    print(y_pred)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()#梯度归零
    loss.backward()#反向传播
    optimizer.step()#Update
print('w=',Model.Linear.weight.item())
print('b=',Model.Linear.bias.item())

x_test=torch.tensor([4.0])
y_test=Model(x_test)
print('y_test.pred=',y_test.item())


import torch
import matplotlib.pyplot as plt
#Step1:DataSet
x_data=torch.tensor([[1.0],[2.0],[3.0]])
y_data=torch.tensor([[3.0],[5.0],[7.0]])

#Step2:Network
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear=torch.nn.Linear(1,1,True)
    def forward(self,x):
        y_pred=self.Linear(x)
        return y_pred
Models={
    'SGD':LinearModel(),
    'Adam':LinearModel(),
    'Adagrad':LinearModel(),
    'Adamax':LinearModel(),
    'ASGD':LinearModel(),
    'RMSprop':LinearModel(),
    'Rprop':LinearModel(),
}
#Step3:Loss and optimizer
criterion=torch.nn.MSELoss(reduction='sum')
optimizer={
    'SGD':torch.optim.SGD(Models['SGD'].parameters(),lr=0.01,momentum=0.8),
    'Adam':torch.optim.Adam(Models['Adam'].parameters(),lr=0.01),
    'Adagrad':torch.optim.Adagrad(Models['Adagrad'].parameters(),lr=0.01),
    'Adamax':torch.optim.Adamax(Models['Adamax'].parameters(),lr=0.01),
    'ASGD':torch.optim.ASGD(Models['ASGD'].parameters(),lr=0.01),
    'RMSprop':torch.optim.RMSprop(Models['RMSprop'].parameters(),lr=0.01),
    'Rprop':torch.optim.Rprop(Models['Rprop'].parameters(),lr=0.01),
}
loss_list={k:[] for k in Models.keys()}
#Step4 :Train:1.前向计算 2.计算loss 3.梯度清零 4.反向传播 5.梯度更新
for opt_name,opt in optimizer.items():
    model=Models[opt_name]
    for epoch in range(100):
        y_pred=model(x_data)
        loss=criterion(y_pred,y_data)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_list[opt_name].append(loss.item())
for opt_name,losses in loss_list.items():
    plt.plot(losses,label=opt_name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt

#制作数据集：
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # ⚠️ 必须加括号
])

train_Data=datasets.MNIST(
    root='D:/dataset/minist/',
    train=True,
    download=True,
    transform=transform
)
test_Data=datasets.MNIST(
    root='D:/dataset/minist/',
    train=False,
    download=True,
    transform=transform
)
#批次加载器：
train_loader=DataLoader(train_Data,shuffle=True,batch_size=256)
test_loader=DataLoader(test_Data,shuffle=False,batch_size=256)

#LeNet-5
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(      #按顺序搭建各层
         nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Tanh(),#C1卷积层
         nn.AvgPool2d(kernel_size=2,stride=2),#S2平均汇聚
         nn.Conv2d(6,16,kernel_size=5),nn.Tanh(),#C3：卷积层
         nn.AvgPool2d(kernel_size=2,stride=2),# S4：平均汇聚
         nn.Conv2d(16,120,kernel_size=5),nn.Tanh(),#C5卷积层
         nn.Flatten(),#展平
         nn.Linear(120,84),nn.Tanh(),#F5：全连接层
         nn.Linear(84,10)#F6全连接层
        )
    def forward(self,x):
        y=self.net(x)
        return y
#查看网络结构：

model=CNN().to('cuda:0')

X=torch.rand(size=(1,1,28,28)).to('cuda:0')
for layer in model.net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)

#loss
criterion=nn.CrossEntropyLoss()
#optim
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

#train
epochs=5
losses=[]

for epoch in range(epochs):
    for x,y in train_loader:
        x,y=x.to('cuda:0'),y.to('cuda:0')
        Pred=model(x)
        loss=criterion(Pred,y)
        losses.append(loss.item())  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

Fig=plt.figure()
plt.plot(range(len(losses)),losses)
plt.show()

model.eval()
correct=0
total=0
with torch.no_grad():
    for x,y in test_loader:
        x,y=x.to('cuda:0'),y.to('cuda:0')
        Pred=model(x) 
        _,predicted=torch.max(Pred.data,dim=1)
        correct+=torch.sum((predicted==y))
        total+=y.size(0)
print('测试正确率：',correct/total*100,'%')
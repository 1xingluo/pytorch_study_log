import torch
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
#创建张量：向量
w=torch.tensor([1.0])
w.requires_grad=True
def forward(x):
    return x*w
#构建流程图
def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2
def mse(xs,ys):
    mse=0
    for x, y in zip(xs, ys):
        l = loss(x, y)      # 计算损失 
        mse+=l
    return mse/len(xs)
print("Before training: 4", forward(4).item())

for epoch in range(1000):
        loss_=mse(x_data,y_data)
        loss_.backward()         # 反向传播：计算 ∂l/∂w
        with torch.no_grad():
            w -= 0.01 * w.grad   #更新梯度，不引入计算图
        w.grad.zero_()           # 梯度清零（否则梯度会累积）
        print('progress',epoch,'w=',w.item(),'loss=',loss_.item())
print("After training: 4", forward(4).item())

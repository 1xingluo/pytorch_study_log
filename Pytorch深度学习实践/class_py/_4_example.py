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
print("Before training: 4", forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)       # 计算损失
        l.backward()         # 反向传播：计算 ∂l/∂w
        w.data = w.data - 0.01 * w.grad.data  # 手动更新参数
        w.grad.zero_()                    # 梯度清零（否则梯度会累积）
    print('progress',epoch,l.item())
print("After training: 4", forward(4).item())

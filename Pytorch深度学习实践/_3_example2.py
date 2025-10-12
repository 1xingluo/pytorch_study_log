import numpy as np
import matplotlib.pyplot as plt
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
w=1
#随机梯度下降，xi,xi+1有依赖，会影响效率，不能并行；
#y_hat=x*w
def forward(x):
    y_pred=x*w
    return y_pred
def loss(x,y):
    loss=(forward(x)-y)**2
    return loss
def gradient(x,y):
    grad=2*x*(forward(x)-y)
    return grad
print('before training',4,forward(4))
w_list=[]
cost_list=[]
for epoch in range(100):
    cost=0
    for x,y in zip(x_data,y_data):
        w-=0.01*gradient(x,y)
        l=loss(x,y)
        cost+=l
    #随机梯度下降中近似表示：
    w_list.append(w)
    cost_list.append(cost/len(x_data))
    print('epoch=',epoch,'cost',cost)
print('after traininig',w,forward(4))
plt.plot(w_list,cost_list)
plt.ylabel('MSE')
plt.xlabel('w')
plt.show()
#引出折中方法，使用Mini_batch:批量梯度下降算法；
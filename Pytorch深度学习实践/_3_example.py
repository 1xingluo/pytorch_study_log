import numpy as np
import matplotlib.pyplot as plt
# 例子：y_hat = x*w 
#loss=(y_hat-y)**2    cost=1/N*∑loss
# 偏L/偏w=1/N ∑2*xn*(xn*w-yn)
# Updata:w=w-α/N*∑2*xn*(xn*w-yn)
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w=1.0
def forward(x):
    y_pred=x*w
    return y_pred
def cost(xs,ys):
    cost = 0
    for x,y in zip(xs,ys):
        loss=(forward(x)-y)**2
        cost+=loss
    return cost/len(xs)
def gradient(xs,ys):
    grad=0
    for x,y in zip(xs,ys):
        grad+=2*x*(forward(x)-y)
    return grad/len(xs)
print('Predict before learning',4,forward(4))
for epoch in range(100):
    cost_val=cost(x_data,y_data)
    grad_val=gradient(x_data,y_data)
    w-=0.01*grad_val
    print('epoch:',epoch,'w=',w,'loss=',cost_val)
print('Predict after training',4,forward(4))

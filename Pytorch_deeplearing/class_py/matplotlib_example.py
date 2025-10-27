import matplotlib.pyplot as plt
import numpy as np

# #1.1绘制图像
# fig1=plt.figure()
# x=[1,2,3,4,5]
# y=[1,8,27,64,125]
# plt.plot(x,y)
# plt.show()

# #1.2保存图片
# fig1.savefig('x^3.svg')

#1.1中为matlab方式
# #1.3面向对象方式：
# x=[1,2,3,4,5]
# y=[1,8,27,64,125]
# fig2=plt.figure()
# axes=plt.axes()
# axes.plot(x,y)
# plt.show()

#二.多图形的绘制

# #2.1绘制多线条
# x=[1,2,3,4,5]
# y1=[2,4,6,8,10]
# y2=[1,4,9,16,25]
# y3=[1,0.5,0.33,0.25,0.2]
# # #matlab方式：
# fig1=plt.figure()
# plt.plot(x,y1)
# plt.plot(x,y2)
# plt.plot(x,y3)
# plt.show()
# #面向对象方式：
# fig2=plt.figure()
# axes=plt.axes()
# axes.plot(x,y1,label='y1=2x')
# axes.plot(x,y2,label='y2=x**2')
# axes.plot(x,y3,label='y3=1/x')
# axes.legend()   
# plt.show()

#2.2绘制多子图
#matlab格式
x=[1,2,3,4,5]
y1=[2,4,6,8,10]
y2=[1,4,9,16,25]
y3=[1,0.5,0.33,0.25,0.2]

# plt.figure()  # 创建 figure
# # 第一个子图
# plt.subplot(3,1,1)  # 3行1列，第1个
# plt.plot(x, y1,'r-o')
# plt.grid(True)
# # 第二个子图
# plt.subplot(3,1,2)  # 3行1列，第2个
# plt.plot(x, y2,'g-s')
# plt.grid(True)
# # 第三个子图
# plt.subplot(3,1,3)  # 3行1列，第3个
# plt.plot(x, y3,'b-^')
# plt.grid(True)

# plt.tight_layout()  # 自动调整子图间距
# plt.show()

#axes方式
# fig1,ax1=plt.subplots(3)
# ax1[0].plot(x,y1,'r-o')
# ax1[1].plot(x,y2,'g-s')
# ax1[2].plot(x,y3,'b-^')
# plt.tight_layout()  # 自动调整子图间距
# plt.show()

 #三 图表类型：

#1.二维图：1.color关键字：2.linestyle 3.linewidth 粗细（0.5-3）4 marker线条的标记：. o ^ s D 5.markersize
# fig1=plt.figure()
# plt.plot(x,y1,color='#b32727',linestyle=':',linewidth=1,marker='D',markersize=6)
# plt.plot(x,y2,color='#403A96',linestyle='--',linewidth=2,marker='o',markersize=10)
# plt.plot(x,y3,color="#87C223",linestyle='-.',linewidth=3,marker='s',markersize=8)
# plt.show()

#2.网格图：
# x1=np.linspace(0,10,1000)
# l=np.sin(x1)*np.cos(x1).reshape(-1,1)
# fig1=plt.figure()
# plt.imshow(l)
# plt.colorbar()
# plt.show()

#3.统计图  keywords：1.bins：2.透明度alpha 3.histtype:使用stepfilled 4.直方图颜色color 5.edgecolor
# data=np.random.randn(10000)
# fig1=plt.figure()
# plt.hist(data,bins=30,alpha=0.5,color="#f9470b",edgecolor='#ffffff')
# plt.show()

#四 图窗属性：

#设置上下限
# fig1=plt.figure()
# plt.plot(x,y2)
# plt.xlim(1,6)
# plt.ylim(1,26)
# plt.show()

#设置label# plt.title  plt.xlabel plt.ylabel
# fig1=plt.figure()
# plt.plot(x,y2)
# plt.title('this is the title')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

#图例 plt.legend()
#fig1=plt.figure()
# plt.plot(x,y1,label='y1=2x')
# plt.plot(x,y2,label='y2=x**2')
# plt.plot(x,y3,label='y3=1/x')
# plt.legend(loc='best',frameon=False,ncol=1)
# plt.show()

#网格：plt.grid()
#fig1=plt.figure()
# plt.plot(x,y1,label='y1=2x')
# plt.plot(x,y2,label='y2=x**2')
# plt.plot(x,y3,label='y3=1/x')
# plt.legend()
# plt.grid()
# plt.show()
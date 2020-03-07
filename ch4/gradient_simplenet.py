# coding: utf-8
#计算神经网络的梯度的示例程序 4.4.2
#即损失函数关于参数的梯度

import sys, os
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) #用高斯分布进行初始化（随机设置权重参数）

    def predict(self, x):
        '''这个网络很明显是一个2层网络'''
        return np.dot(x, self.W)

    def loss(self, x, t):
        '''x接收输入数据 t接收正确解标签'''
        z = self.predict(x)
        y = softmax(z) #激活函数转换
        loss = cross_entropy_error(y, t)
        return loss #返回误差

#f = lambda w: net.loss(x, t)
def f(fake_parameter):
	#伪参数,因为numerical_gradient(f, x)会在内部执行 f(x)，为了与之兼容而定义
	return net.loss(x,t) #返回损失函数（误差）


x = np.array([0.6, 0.9]) #输入数据
t = np.array([0, 0, 1]) #正确解标签

net = simpleNet()

dW = numerical_gradient(f, net.W) #计算损失函数关于参数的梯度，即偏导数
print(dW)

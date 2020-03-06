# coding: utf-8

import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from common.functions import sigmoid, identity_function
import numpy as np

def init_network():
	'''初始化网络'''
	network = {} # 字典
	network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) #第0层和第1层之间的权重参数
	network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) #第1层和第2层之间的
	network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]]) #第2层和第3层（输出层）之间的
	network['b1'] = np.array([0.1, 0.2, 0.3]) #偏置
	network['b2'] = np.array([0.1, 0.2])
	network['b3'] = np.array([0.1, 0.2])
	return network

def forward(network, x):
	'''前向传播'''
	W1, W2, W3 = network['W1'], network['W2'], network['W3'] #从字典取出参数
	b1, b2, b3 = network['b1'], network['b2'], network['b3']

	a1 = np.dot(x, W1) + b1 #利用点积进行从输入层到第1层的信号传递
	z1 = sigmoid(a1) #使用激活函数转换第1层的输出
	a2 = np.dot(z1, W2) + b2 #同上
	z2 = sigmoid(a2) #同上
	a3 = np.dot(z2, W3) + b3
	y = identity_function(a3) #这里使用恒等函数，只是为了和上面保持格式一致
	
	return y #返回网络的计算结果

network = init_network() #初始化网络
x = np.array([1.0, 0.5]) #输入数据
y = forward(network, x) #将输入数据输入网络，并返回输出数据
print(y) #0.31682708 0.69627909

# coding: utf-8
#计算手写数字识别的二层神经网络模型的精度，
#参数和偏置是已经准备好的

import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np 
import pickle # 保存了权重参数的文件
from dataset.mnist import load_mnist #从该数据集中导入测试数据和正确解标签
from common.functions import sigmoid, softmax #导入softmax等函数

#sys.path.append(os.pardir) 语句实际上是把父目录 deep-learning-from-scratch 
#加入到 sys.path（Python 的搜索模块的路径集）中，
#从而可以导入 deep-learning-from-scratch 下的任何目录（包括 dataset 目录）中的任何文件

def get_data():#获取数据集中的数据
    (x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():  #读入保存在 pickle 文件 sample_weight.pkl 中的学习到的权重参数
    with open("sample_weight.pkl", 'rb') as f:  #这个文件中以字典变量的形式保存了权重和偏置参数
        network = pickle.load(f)
    return network


def predict(network, x): #模型的推理
    W1, W2, W3 = network['W1'], network['W2'], network['W3']#取出权重参数
    b1, b2, b3 = network['b1'], network['b2'], network['b3']#取出偏置

    a1 = np.dot(x, W1) + b1	#输入到第一层的计算
    z1 = sigmoid(a1) #使用激活函数非线性化
    a2 = np.dot(z1, W2) + b2 #第一层到第二层的计算
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3 #第二层到输出层
    y = softmax(a3) #使用softmax处理输出，转换成了概率

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 获取概率最高的元素的索引
    if p == t[i]: #如果这个索引和正确解标签吻合
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))#计算识别精度


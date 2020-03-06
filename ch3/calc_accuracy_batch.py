# coding: utf-8
# 计算3层神经网络模型的精度
# 基于批处理：简单来说，就是一次处理一批图片，整体运算速度更快

import sys, os
sys.path.append(os.pardir)  #为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, \
        flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100 #设置批处理的规模，一次处理100张
accuracy_cnt = 0

for i in range(0, len(x), batch_size): 
    x_batch = x[i:i+batch_size] #以100为单位将数据提取为批数据
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)  #找到第0维中值最大的元素的索引（在这里就相当于所计算的概率最大的数字）
    accuracy_cnt += np.sum(p == t[i:i+batch_size])  #和解标签比较

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

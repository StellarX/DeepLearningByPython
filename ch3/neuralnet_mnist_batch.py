# coding: utf-8
#计算手写数字识别的二层神经网络模型的精度，
#参数和偏置是已经准备好的
'''基于批处理的代码实现'''

import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
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

batch_size = 100 # 批数量
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size] #以100为单位将数据提取为批数据
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)  #找到第0维中值最大的元素的索引（在这里就相当于所计算的概率最大的数字）
    accuracy_cnt += np.sum(p == t[i:i+batch_size])  # 把这些数字和测试数据比较 这里应该是只要有一个不同等号就不成立了

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

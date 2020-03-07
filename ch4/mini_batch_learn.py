
#本程序实现以下功能：（这个程序书上没有完整示例，自己简单写了一下）
#
#计算三层神经网络模型(手写数字识别)的精度（基于mini-batch）
#计算交叉熵误差（基于mini-batch，且t为标签形式）(待实现)
#
#mini-batch：简单来说就是：随机抽取小批量数据近似所有数据

import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle #保存了权重参数的文件
from common.functions import sigmoid, softmax, cross_entropy_error_2 #导入softmax等函数
# import common.functions as f
from dataset.mnist import load_mnist


def init_network():  
    '''读入保存在 pickle 文件 sample_weight.pkl 中的学习到的权重参数,
    这个文件中以字典变量的形式保存了权重和偏置参数'''
    with open("../ch3/sample_weight.pkl", 'rb') as f: #以二进制方式读取
        network = pickle.load(f)
    return network


def predict(network, x): 
    '''模型的推理(前向传播)'''
    W1, W2, W3 = network['W1'], network['W2'], network['W3'] #取出权重参数
    b1, b2, b3 = network['b1'], network['b2'], network['b3'] #取出偏置

    a1 = np.dot(x, W1) + b1	#输入（第0层）到第1层的计算
    z1 = sigmoid(a1) #使用激活函数非线性化
    a2 = np.dot(z1, W2) + b2 #第1层到第2层的计算
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3 #第2层到输出层
    y = softmax(a3) #使用softmax处理输出，转换成了概率

    return y


(x_train, t_train), (x_test, t_test) = \
	load_mnist(normalize=True, flatten=True, one_hot_label=False) #注意不是one-hot表示

train_size = x_test.shape[0]
mini_batch_size = 100
temp = np.random.choice(train_size, mini_batch_size) #获取随机批量数据在原数据集的索引
x_batch = x_test[temp] #根据索引取出数据
t_batch = t_test[temp] #同上

# output = np.array(10)

x = x_batch
t = t_batch
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
	y = predict(network, x[i])
	p = np.argmax(y) #获取概率最高的元素的索引
	if p == t[i]: #如果这个索引和正确解标签吻合
	    accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x))) #计算识别精度 0.9352

# print(output)
# ans = cross_entropy_error_2(output, t_batch)
# print(ans)




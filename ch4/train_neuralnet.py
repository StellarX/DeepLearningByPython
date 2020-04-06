# coding: utf-8
'''进行二层网络的学习，每次更新参数后，并将训练数据和测试数据的识别精度分别显示出来
可以看到，这两个值都在增加，并且基本吻合，说明没有发生过拟合
'''

import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

#超参数
iters_num = 10000  # 适当设定梯度法循环的次数，也就是更新参数的次数
train_size = x_train.shape[0] #60000
batch_size = 100
learning_rate = 0.1

train_loss_list = [] #保存每次更新参数后的损失函数的值
train_acc_list = [] #保存每次经过1个epoch后所计算的训练数据的识别精度
test_acc_list = [] #同上，只不过是测试数据

iter_per_epoch = max(train_size / batch_size, 1) #表示更新多少次参数才把训练数据“看完”  600
#epoch是一个单位  一个epoch表示学习中所有训练数据均被使用过一次时的更新次数

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) #从60000个训练数据中随机取出100个
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 计算梯度
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch) 
    
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch) #计算损失函数的值
    train_loss_list.append(loss)
    
    # 计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train) #（整个）训练数据的精度
        test_acc = network.accuracy(x_test, t_test) #（整个）测试数据的精度
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(i)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 绘制图形
# markers = {'train': 'o', 'test': 's'} #?
# x = np.arange(len(train_acc_list))
# plt.plot(x, train_acc_list, label='train acc')
# plt.plot(x, test_acc_list, label='test acc', linestyle='--')
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.ylim(0, 1.0) #设定y轴坐标范围
# plt.legend(loc='lower right') #设定图例的位置 右下方
# plt.show()

x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list)
plt.xlabel("x")
plt.ylabel("loss")
plt.ylim(0, 9)
plt.legend()
plt.show()
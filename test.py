
#本程序实现以下功能：（这个程序书上没有完整示例，自己简单写了一下）
#
#计算三层神经网络模型(手写数字识别)的精度（基于mini-batch）
#计算交叉熵误差（基于mini-batch，且t为标签形式）
#
#mini-batch：简单来说就是：随机抽取小批量数据近似所有数据

import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle #保存了权重参数的文件
from common.functions import sigmoid, softmax #导入softmax等函数
# import common.functions as f
from dataset.mnist import load_mnist

x = np.arange(0, 100, 10).reshape(2, 5)
print(x.size)
grad = np.zeros_like(x)
print(grad)
it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

# while not it.finished:
# 	print(it.multi_index)
# 	it.iternext()

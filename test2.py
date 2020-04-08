
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import json

from utils import mnist_reader
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)
batch_mask = np.random.choice(60000, 100)

x_batch = x_train[batch_mask]
print(x_batch.shape)
###################################待记忆
x = np.random.rand(2,3)
# print(x)

x = [3,5,67,3,5,3,9]
print(x[:3])
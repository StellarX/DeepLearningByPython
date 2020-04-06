
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

from utils import mnist_reader
mnist_train = mnist_reader.load_mnist('dataset2/fashion', kind='train') #return two arrays
mnist_test = mnist_reader.load_mnist('dataset2/fashion', kind='t10k')


x = np.array([[51, 55], [14, 19], [0, 4]]) 
for t in x:
	print(t)


		
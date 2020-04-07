
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import json

from utils import mnist_reader
mnist_train = mnist_reader.load_mnist('dataset2/fashion', kind='train') #return two arrays
mnist_test = mnist_reader.load_mnist('dataset2/fashion', kind='t10k')


with open("1.txt", "r") as f_obj:
	c = f_obj.read()

dic = {
	"jack":1,
	"tom":2,
	"mary":3
}



num = "json datas"
with open("json.txt", "w") as f_obj:
	json.dump(num, f_obj)

with open("json.txt", "r") as f_obj:
	r = json.load(f_obj)
print(r)
		

###################################待记忆
x = np.random.rand(2,3)
# print(x)

t = np.array([[1,4,7],[2,3,4]])
t = t.T
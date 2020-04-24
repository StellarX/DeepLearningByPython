import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from utils import mnist_reader
import math
# from sklearn import preprocessing 

def calc_mean(n, m, img):
	mean = []
	for i in range(n):
		sum = 0
		for j in range(m):
			sum += img[i][j]
		mean.append(float(sum/784))
	return mean

def calc_std(n, m, img, mean):
	std = []
	for i in range(n):
		sum = 0
		for j in range(m):
			sum += (img[i][j] - mean[i])**2
		std.append(math.sqrt(float(sum/784)))
	return std

def transfer_z_score(n, m, img, mean, std):
	img_z_score = [None]*n
	for i in range(len(img_z_score)): # 创建数组
	    img_z_score[i] = [0]*m

	for i in range(0, n):
		img_z_score[i] = (img[i] - mean[i]) / std[i]
	return img_z_score



mnist_train = mnist_reader.load_mnist('../dataset2/fashion', kind='train') #kind ???
mnist_test = mnist_reader.load_mnist('../dataset2/fashion', kind='t10k')

labels_map2 = ['T恤', '裤子', '套衫', '连衣裙', '外套',
               '凉鞋', '衬衫', '运动鞋', '背包', '短靴']
labels_map = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']  

img, label = mnist_train
img2, label2 = mnist_test
img = np.array(img)
img2 = np.array(img2)
n, m = 60000, 784
images = img

print(images.shape)


# 对mnist_train进行预处理
mean = np.mean(images, axis=1) #计算均值
# mean = calc_mean(n, m, img2)
std = np.std(images, axis=1) #计算标准差
# std = calc_std(n, m, img2, mean)
img_z_score = transfer_z_score(n, m, images, mean, std) #Z_score归一化



plt.figure() # 创建画板
for i in range(1,11):
    plt.subplot(3,4,i)
    plt.xlabel(labels_map[label[i-1]]) #绘制标签
    
    plt.imshow(img_z_score[i-1].reshape(28, 28)) #恢复图像的形状
    plt.xticks([]) #去除坐标轴
    plt.yticks([])
plt.show()

  
print(np.mean(img_z_score[0]))
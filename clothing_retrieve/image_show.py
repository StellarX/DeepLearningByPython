import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from utils import mnist_reader


mnist_train = mnist_reader.load_mnist('../dataset2/fashion', kind='train') #kind ???
mnist_test = mnist_reader.load_mnist('../dataset2/fashion', kind='t10k')


labels_map2 = ['T恤', '裤子', '套衫', '连衣裙', '外套',
               '凉鞋', '衬衫', '运动鞋', '背包', '短靴']
labels_map = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']  

img, label = mnist_train[0:100]


plt.figure() # 创建画板
for i in range(1,101):
    plt.subplot(10,10,i)
    # plt.xlabel(labels_map[label[i-1]]) #绘制标签
    
    plt.imshow(img[i-1].reshape(28, 28)) #恢复图像的形状
    plt.xticks([]) #去除坐标轴
    plt.yticks([])
plt.show()

# print(img[0].reshape(28,28))

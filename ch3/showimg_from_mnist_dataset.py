
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
	pil_img = Image.fromarray(np.uint8(img)) 
	#将保存为numpy数组的图像数据转化为PIL用的数据对象
	pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, 
	normalize = False)

img = x_train[0]
label = t_train[0]

# print(label)
print(img.shape)
img = img.reshape(28, 28)#把图像变为原来的尺寸
# img = img.reshape(14, 56)
print(img.shape)

img_show(img)
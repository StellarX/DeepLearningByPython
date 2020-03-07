'''
梯度下降法求函数的极小值 示例程序
'''
import sys, os
sys.path.append(os.pardir)  #为了导入父目录的文件而进行的设定
import numpy as np 
from common.gradient import numerical_gradient_0, numerical_gradient


def function_2(x):
	return x[0]**2 + x[1]**2

def gradient_descend(f, init_x, learn_rate, step):
	'''使用梯度下降法求函数的极小值（注意不是最小值，可能为最小值）'''
	x = init_x
	for i in range(step):
		grad = numerical_gradient_0(f, x)
		x = x - learn_rate * grad
	return x


init_x = np.array([-3.0, 4.0])
ans = gradient_descend(function_2, init_x, 0.1, 100) #设定学习率0.1 迭代次数100次
print(ans)
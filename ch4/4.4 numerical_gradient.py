import matplotlib.pylab as plt
import numpy as np


def function_2(x):
	return np.sum(x**2)

#4.4 梯度实现（由全部变量的偏导数汇总而成的向量）
#实际上就是对f的各个变量求偏导
def numerical_gradient(f,x):
	h = 1e-4
	grad = np.zeros_like(x) #生成和形状相同的数组
	
	for idx in range(x.size):
		tmp_val = x[idx]
		#f(x+h)
		x[idx] = tmp_val + h
		fxh1 = f(x)
		
		#f(x-h)
		x[idx] = tmp_val - h
		fxh2 = f(x)
		
		grad[idx] = (fxh1 - fxh2) / 2*h
		x[idx] = tmp_val #还原值
		
	return grad

out1 = numerical_gradient(function_2, np.array([3.0 , 4.0]))
#out2 = numerical_gradient(function_2, np.array([]))
print(out)





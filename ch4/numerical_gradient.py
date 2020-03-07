
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.gradient import numerical_gradient_0 #导入求梯度的程序

def function_2(x):
	return np.sum(x**2)

out = numerical_gradient_0(function_2, np.array([3.0 , 4.0]))
print(out)
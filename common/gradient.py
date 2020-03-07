# coding: utf-8
import numpy as np


def _numerical_gradient_1d(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        
    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad


def numerical_gradient(f, x):
    '''数值梯度：版本2 为了接收多维numpy数组而改进'''
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite']) #使用迭代器遍历
 
    while not it.finished:
        idx = it.multi_index #取出当前的元素的索引
        tmp_val = x[idx]
        
        x[idx] = float(tmp_val) + h #为什么下面不加float呢？？
        fxh1 = f(x)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 还原值
        it.iternext() #进入下一次迭代
    return grad

def numerical_gradient_0(f, x):
    '''梯度实现(版本0) x为一维numpy数组(4.4)  这个接收不了多维数组，因为idx那里'''
    h = 1e-4
    grad = np.zeros_like(x) #生成和x形状相同、所有元素为0的数组
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val #还原值
    return grad


def numerical_diff_0(f, x):
    '''利用中心差分求数值导数（4.3.1）'''
    h = 1e-4
    return (f(x + h) - f(x - h)) / 2*h



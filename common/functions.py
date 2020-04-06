# coding: utf-8
import numpy as np


def identity_function(x):
    return x

#阶跃函数
def step_function(x):
    return np.array(x > 0, dtype=np.int)

#sigmoid激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    
#RELU激活函数
def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    
#输出层softmax函数，用于分类问题，用在模型的学习阶段
def softmax(x):
    if x.ndim == 2:	#如果是二维矩阵
        x = x.T	#转置  ？
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 防止溢出
    return np.exp(x) / np.sum(np.exp(x))

def softmax_0(x):
    '''输出层的激活函数: 版本0'''
    c = np.max(x)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)

    return exp_a / sum_exp_a


def mean_squared_error(y, t):
    '''均方误差(MSE)'''
    return 0.5 * np.sum((y-t)**2)





'''以下是计算交叉熵误差的函数，有几个版本，从易到难'''
#计算交叉熵误差   需要再理解一下程序
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def cross_entropy_error_2(y, t):
    '''交叉熵误差：版本2：计算批量数据的误差
    但t为标签形式'''
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    # print(batch_size)
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def cross_entropy_error_1(y, t):
    '''交叉熵误差：版本1：增加计算批量数据（mini-batch）的误差
    t为one-hot表示'''
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y+1e-7)) / batch_size

def cross_entropy_error_0(y, t):
    '''交叉熵误差：版本0，计算单个数据的
    t为one-hot表示'''
    delta = 1e-7 #防止y=0时log计算结果变成-inf
    return -np.sum(t * np.log(y+delta))






def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

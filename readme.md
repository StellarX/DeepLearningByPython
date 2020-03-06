# 文件夹和文件说明
## ch3
- three_layer_net:三层神经网络的推理过程简单实现(3.4)
- showimg_from_mnist_dataset：从minit数据集读取第一个图片并显示（3.6.1）
- calc_accuracy.py:计算手写数字识别的三层神经网络模型的精度（3.6.2）
- calc_accuracy_batch.py:同上，只不过基于批处理（3.6.3）
- sample_weight.pkl:以字典变量的形式保存了权重和偏置参数

## ch4
- numerical_diff.py:计算数值微分的程序(4.3)
- gradient_2d.py:绘制梯度的程序(4.4)
- numerical_gradient.py:求梯度的程序(4.4)
- gradient_simplenet.py:计算神经网络的梯度的程序(4.4.2)
- two_layer_net.py:创建一个二层神经网络的类(4.5.1) 
- train_neuralnet.py:进行二层网络的学习/训练(4.5.2)

## common
- function.py:基本函数模块
- gradient.py:

## dataset:训练和测试数据集

### 注意
- 所有网络的层数不包括输出层，即如果一个网络为：输入、隐藏层1、隐藏层2、输出，则该网络为3层网络，实际上输入层称为第0层
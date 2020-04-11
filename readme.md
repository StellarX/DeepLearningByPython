# 文件夹和文件说明
## ch3
- three_layer_net: 三层神经网络的推理过程简单实现(3.4)
- showimg_from_mnist_dataset： 从minit数据集读取第一个图片并显示（3.6.1）
- calc_accuracy.py: 计算手写数字识别的三层神经网络模型的精度（3.6.2）
- calc_accuracy_batch.py: 同上，只不过基于批处理（3.6.3）
- sample_weight.pkl: 以字典变量的形式保存了权重和偏置参数

## ch4
- mini_batch_learn.py: 基于小批量的计算手写数字识别的三层神经网络模型的精度
- numerical_diff.py: 计算数值微分的程序(4.3)
- numerical_gradient.py: 求梯度（4.4.0）
- gradient_descend.py: 梯度下降法求函数极小值（4.4.1）
- gradient_2d.py: 绘制梯度的程序(4.4)
- gradient_simplenet.py: 计算神经网络的梯度的程序(4.4.2)
- two_layer_net.py: 创建一个二层神经网络的类(4.5.1) 
- train_neuralnet.py: 进行二层网络的学习/训练，并绘制识别精度变化图(4.5.2)

## ch5
- AddLayerAndMulLayer.py: 加法层和乘法层的封装类（包含反向和正向传播）
- TwoLayerNet.py: 使用了误差反向传播的二层网络模型，且每一层的运算函数都封装为了类
- gradient_check.py: 使用数值梯度检查误差反向传播求出的梯度是否存在较大误差
- train_net.py: 进行二层网络的学习，其中使用了误差反向传播法，且封装了各层处理，速度更快(怎么感觉也没快多少。。)

## ch6
- optimizer_compare_naive.py: 将4种参数更新方法SGD、Momentum、AdaGrad、Adam通过图像直观展示出来
- optimizer_compare_mnist.py: 分别使用4种参数更新方法来进行手写数字识别，并使用图像展示学习的差异

## clothing_retrieve（服装检索代码）
- image_show.py:读取数据集并显示图片
- netbug1.py:爬虫
- z_score_preprocessing.py:数据集预处理

## common
- function.py: 基本函数模块
- gradient.py: 求梯度的几种函数
- Layers.py: 将Relu、sigmoid的处理封装为类（层）；将加权和、加偏置封装为Affine层（类）；将softmax和交叉熵误差封装为SoftmaxWithLoss层（类）

## dataset: mnist数据集
## dataset2: fashion-mnist数据集
## utils:fashion-mnist:数据集的读取模块等。。。

#### 注意
- 所有网络的层数不包括输出层，即如果一个网络为：输入、隐藏层1、隐藏层2、输出，则该网络为3层网络，实际上输入层称为第0层
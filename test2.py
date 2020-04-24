
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import json
import cv2


def snape(xiaochi):
	while "apple" in xiaochi:
		xiaochi.remove("apple")
	print(xiaochi)


a = ["jack", "tom", "mary", "linda"]
b = ["apple", "peel", "orange"]



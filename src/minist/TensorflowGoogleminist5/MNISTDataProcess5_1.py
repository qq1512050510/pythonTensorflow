#codeing:utf-8
'''
Created on Mar 5, 2019

@author: jyp
'''
from tensorflow.examples.tutorials.mnist import input_data
#import input_data
mnist = input_data.read_data_sets("../../minist/origin/",one_hot=True)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import numpy as np


#查看训练数据大小
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
#查看验证数据大小
print(mnist.validation.images.shape)
print(mnist.validation.labels.shape)
#查看测试数据大小
print(mnist.test.images.shape)
print(mnist.test.labels.shape)

#打印Training data size: 55000.
print("Training data size:",mnist.train.num_examples)

#打印Validating data size: 5000
print("Validating data size:",mnist.validation.num_examples)

#打印Testing data size:1000
print("Testing data size:",mnist.test.num_examples)

#打印Example training data:[0. 0. 0. ....]
print("Example training data:",mnist.train.images[0])

print(mnist.train.images.shape)
print(mnist.train.images[0].shape)
print(28*28)
print(range(28))

for i in range(28):
    #print(np.around(mnist.train.images[0][i*28:i*28+28],decimals=1));
    print(np.around(mnist.train.images[0][i*28:i*28+28]));

#打印Example training data label:
print("Example training data label:",mnist.train.labels[0])











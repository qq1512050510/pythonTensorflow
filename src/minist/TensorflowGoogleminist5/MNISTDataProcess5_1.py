#codeing:utf-8
'''
Created on Mar 5, 2019

@author: jyp
'''
from tensorflow.examples.tutorials.mnist import input_data
#import input_data
mnist = input_data.read_data_sets("../../minist/",one_hot=True)
#查看训练数据大小
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
#查看验证数据大小
print(mnist.validation.images.shape)
print(mnist.validation.labels.shape)
#查看测试数据大小
print(mnist.test.images.shape)
print(mnist.test.labels.shape)

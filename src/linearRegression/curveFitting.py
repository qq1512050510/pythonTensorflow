#-*- coding:utf-8 -*-
'''
Created on 2019年1月22日
@url https://segmentfault.com/a/1190000017970663
@author: adp
'''
import numpy as np
import tensorflow as tf

import matplotlib
#from cffi.backend_ctypes import xrange  
# compatibility mac
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

# 使用 NumPy 生成 模拟数据（phony data），总共 100个点
x_data = np.float32(np.random.rand(2,100))
print(x_data)
y_data = np.dot([0.100,0.200],x_data) + 0.300
print(y_data)


#构造一个线性模型

b = tf.Variable(tf.zeros([1]))
print(b)
W = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))
print(W)
#矩阵相乘
y = tf.matmul(W,x_data) + b
print(y)

# 最下化方差
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#初始化变量，旧函数（initialize_all_variables）已经废弃，替换为新函数
init = tf.global_variables_initializer()

#启动图（graph）
sess = tf.Session()
sess.run(init)


#拟合平面
for step in range(0,201): 
    sess.run(train)
    if step%20 == 0 :
        print(step,sess.run(W),sess.run(b))








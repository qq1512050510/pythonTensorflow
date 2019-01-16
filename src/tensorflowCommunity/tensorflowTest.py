#-*- coding:utf-8 -*-
'''
Created on 2019年1月15日

@author: adp
'''
import tensorflow as tf
import numpy as np

#使用 Numpy 生成假数据（phony data),总共100个点
x_data = np.float32(np.random.rand(2,100))
print(x_data)
y_data = np.dot([0.100,0.200],x_data)+0.300
print(y_data)
print(x_data.shape)
print(y_data.shape)

#构造一个线性模型
#
b = tf.Variable(tf.zeros([1]))
print(b)
w = tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y = tf.matmul(w, x_data) + b


# 最小化方差
loss  = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


# 初始化变量

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer();


#启动图（graph）

sess = tf.Session()
sess.run(init)

#拟合平面

for step in range(0,201):
    print (step,sess.run(w),sess.run(b))
    sess.run(train)
    #if step % 20 == 0 :
    print (step,sess.run(w),sess.run(b))












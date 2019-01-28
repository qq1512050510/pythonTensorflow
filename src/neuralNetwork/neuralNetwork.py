#-*- coding:utf-8 -*-
'''
Created on Jan 29, 2019
book url: 链接:https://pan.baidu.com/s/1JtinvJ1SuAMDtfrWuW72pg  密码:g170
Page:62
@author: jyp
'''
import tensorflow as tf 
#Numpy是一个科学计算的工具包，这里通过Numpy工具生成模拟数据集
from numpy.random import RandomState

#定义训练数据集batch的大小
batch_size = 8

#定义神经网络的参数，这里还是沿用 3.4.2 小节中给出的神经网络结构
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
with tf.Session() as sess:
    print(sess.run(w1.initializer))
    print(sess.run(w1))
w2 = tf.Variable(tf.random_normal([3,2],stddev=1,seed=1))



#在shape的一个维度上使用None可以方便使用不打的batch大小。在训练时需要把数据分成
#比较小的batch，但是在测试时，可以一次性使用全部的数据。当数据集比较小时这样比较
#方便测试，但数据集比较大时，将大量数据放入batch可能导致内存溢出

x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')

#定义神经网络前向传播的过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#定义损失函数和反向传播的算法
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


    


# -*- coding:utf-8 -*-
'''
Created on 2018年9月30日

@author: adp
'''
impimport tensorflowTest as tfmport numpy as np
from tensorflow.contrib.learn.python.learn.graph_actions import train

# 使用Numpy 生成数据（phony data），总共100个点
x_data = np.float32(np.random.rand(2,100))
print (x_data) 
#print (np.random.rand(2,100))

y_data = np.dot([0.100,0.200],x_data)+0.300
print ("-------")
print (y_data)


b = tf.Variable(tf.zeros([1]))
print (b)
W = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))
print (W)
y = tf.matmul(W,x_data) +b
print (y)

#最小化方差
loss = tf.reduce_mean(tf.square(y-y_data))
print(loss)
optimizer = tf.train.GradientDescentOptimizer(0.5)
print(optimizer)
init = tf.global_variables_initializer()


#启动图（graph）
sess = tf.Session()
sess.run(init)


# 拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(W), sess.run(b))
        
    


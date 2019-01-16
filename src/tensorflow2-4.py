#-*- coding:utf-8 -*-
'''
Created on 2019年1月11日

@author: adp
'''
import tensorflow as tf

m1 = tf.constant([[1.,2.]])
m2 = tf.constant([[1],[2]])
m3 = tf.constant([[[1,2],
                   [3,4],
                   [5,6],
                   [7,8],
                   [9,10],
                   [11,12]]])
x = tf.constant([[1,2]])
neg_x = tf.negative(x)
session = tf.Session();
x_op = tf.constant([1,2])
neg_op = tf.negative(x_op)
with tf.Session() as sess:
    result = sess.run(neg_op)
    resultx = sess.run(x_op)

print('test')
print(result)
print(resultx)
print('test')
    
print(session.run(m1))
print(neg_x)
print(m1)
print(m2)
print(m3)
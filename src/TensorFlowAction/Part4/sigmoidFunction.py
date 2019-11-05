#coding:utf-8
'''
Created on Mar 19, 2019

@author: jyp
'''
import tensorflow as tf
a = tf.constant([[1.0,2.0],[1.0,2.0],[1.0,2.0]])
sess = tf.Session()
print(sess.run(tf.sigmoid(a)))




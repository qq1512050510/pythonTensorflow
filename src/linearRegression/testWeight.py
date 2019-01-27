#-*- coding:utf-8 -*- 
'''
Created on Jan 27, 2019

@author: jyp
'''
import tensorflow as tf 

weights = tf.Variable(tf.random_normal([2,3],stddev=2))

sess = tf.Session()
init = tf.global_variables_initializer()
print(sess.run(init))
print(sess.run(weights))

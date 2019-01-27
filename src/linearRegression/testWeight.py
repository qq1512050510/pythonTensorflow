#-*- coding:utf-8 -*- 
'''
Created on Jan 27, 2019

@author: jyp
'''
import tensorflow as tf 

weights = tf.Variable(tf.random_normal([2,3],stddev=2))
with tf.Session() as sess:
    print(sess.run(weights))
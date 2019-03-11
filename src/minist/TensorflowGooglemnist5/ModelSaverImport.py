#coding:utf-8
'''
Created on Mar 11, 2019

@author: jyp
'''
import tensorflow as tf
v1 = tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
v2 = tf.Variable(tf.constant(1.0,shape=[1]),name="v2")
result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,"./ModelSaver/model.ckpt")
    print(sess.run(result))
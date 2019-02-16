#coding:utf-8
'''
Created on Feb 16, 2019

@author: jyp
'''
import tensorflow as tf
from numpy import dtype
v = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])
print(tf.clip_by_value(v, 2.5, 4.5))
with tf.Session() as sess:
    print(sess.run(tf.clip_by_value(v, 2.5, 4.5)))
    print(sess.run(tf.log(v)))
    m = tf.clip_by_value(v, 2.5, 4.5)
    print(m.eval())
    print(tf.clip_by_value(v, 2.5, 4.5).eval())
    x=[[1,2,3],[1,2,3]]
    xx = tf.cast(x, tf.float32)
    print(xx.eval())
    print("test")
    print(tf.reduce_mean(xx,0).eval())
    print(tf.reduce_mean(xx,1).eval())
    print(tf.reduce_sum(xx,0).eval())
    y = tf.constant([0.5,0.4,0.1])
    y_ = tf.constant([1,0,0])
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y,y_)
    
    
    #function tf.select tf.greater
    v1 = tf.constant([1.0,2.0,3.0,4.0])
    v2 = tf.constant([4,3,2,1],dtype=tf.float32)
    print(type(v1[0]))
    print(tf.greater(v1,v2).eval())
    
    print((tf.where(tf.greater(v1,v2),v1,v2)).eval())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

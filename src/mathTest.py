#-*- coding:utf-8 -*-
'''
Created on 2019年1月11日

@author: adp
'''
from math import pi
import tensorflow as tf
mean = 0.0
sigma = 1.0
x = 56
neg_x = tf.negative(x)
print(tf.exp(tf.negative(tf.pow(x-mean,2.0)/(2.0*tf.pow(sigma,2.0) )))
*(1.0/(sigma*tf.sqrt(2.0*pi))))

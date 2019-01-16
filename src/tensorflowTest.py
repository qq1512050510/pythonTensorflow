#-*-coding:utf-8 -*-
'''
Created on 2019年1月10日

@author: adp
'''
#import tensorflow as tf
import tensorflow as tf
import numpy as np

from numpy.core.multiarray import dtype


m1 = [[1.0,2.0],
      [3.0,4.0]]
m2 = np.array([[1.0,2.0],[3.0,4.0]],dtype=np.float32)
m3 = tf.constant([[1.0,2.0],[3.0,4.0]])

#print(m3)
if __name__=='__main__':
    print(type(m1))
    print(type(m2))
    print(type(m3))
    print(np.float32)
    
    t1 = tf.convert_to_tensor(m1,dtype=tf.float32)
    t2 = tf.convert_to_tensor(m2,dtype=tf.float32)
    t3 = tf.convert_to_tensor(m3,dtype=tf.float32)
    
    print(t1)
    print(type(t1))
    print(t2)
    print(type(t2))
    print(t3)
    print(type(t3))
    
    
    
    
    print(m2)
    #m3 = tf.constant([[1.0,2.0],[3.0,4.0]])
    print(m1); 
    print('你好');
    pass
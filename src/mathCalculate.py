# -*-coding:utf-8 -*-
'''
Created on 2018年9月30日

@author: adp
'''

import numpy as np
#import matplotlib as plot
#import matplotlib.pyplot as plt
import tensorflow as tf
x = tf.constant([[ 1 ,  2 ]]) #A
neg_x  = tf.negative(x) #B
print (neg_x) #B

x = np.linspace(0,2*np.pi,10)

print(x.shape)
print (x)
#plt.plot(x,np.sin(x))
#plt.show()

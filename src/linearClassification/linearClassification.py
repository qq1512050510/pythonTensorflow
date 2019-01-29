# -*- coding:utf-8 -*-
'''
Created on 2019年1月29日

@author: adp
'''
import tensorflow as tf
import numpy as np

import matplotlib
from linearRegression.linearRegression3_2 import cost
# mac
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

x_label0 = np.random.normal(5,1,10)
x_label1 = np.random.normal(2,1,10)
xs = np.append(x_label0,x_label1)
labels = [0.]*len(x_label0) + [1.]*len(x_label1)

plt.scatter(xs,labels)
plt.show();

learning_rate = 0.001
training_epochs = 1000

X = tf.placeholder("float")
Y = tf.placeholder("float")

def model(X,w):
    return tf.add(tf.multiply(w[1],tf.pow(X,1)),
                  tf.multiply(w[0],tf.pow(X,0)))
    
w = tf.Variable([0.,0.],name="parameters")
y_model = model(X,w)
cost = tf.reduce_sum(tf.square(Y-y_model))

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)












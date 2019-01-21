#-*- coding:utf-8 -*-
'''
Created on 2019年1月21日

@author: adp
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.core.framework import cost_graph_pb2

learning_rate = 0.01
training_epochs = 100
x_train = np.linspace(-1,1,101)
randNp = np.random.rand(*x_train.shape)*0.33
#print(randNp)
#print(x_train.shape)
#print(*x_train)
#print(x_train)
#print(np.random.rand(*x_train.shape))
y_train = 2 * x_train + np.random.rand(*x_train.shape)*0.33

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


def model(X,w):
    return tf.multiply(X,w)

w = tf.Variable(0.0,name="weights")

y_model = model(X,w)
cost = tf.square(Y-y_model)

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer();
    sess.run(init)
    
    for epoch in range(training_epochs):
        for (x,y) in zip(x_train,y_train):
            sess.run(train_op,feed_dict={X:x,Y:y})
            
    w_val = sess.run(w)

print(w_val)
plt.scatter(x_train,y_train)
y_learned = x_train*w_val 
plt.plot(x_train,y_learned,'y')
plt.show()


















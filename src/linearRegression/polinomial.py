#-*- coding:utf-8 -*-
'''
Created on 2019年1月23日

@author: adp
'''
import tensorflow as tf
import numpy as np

import matplotlib
# compatibility mac
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


learning_rate = 0.01
training_epochs = 40


trX = np.linspace(-1,1,101)

num_coeffs = 6
trY_coeffs = [1,2,3,4,5,6]
trY = 0

for i in range(num_coeffs):
    print(trY_coeffs[i])
    trY += trY_coeffs[i] * np.power(trX,i)
    print(trY)
print(trY)
print(trY.shape)

trY += np.random.rand(*trX.shape)

plt.scatter(trX,trY)
plt.show()

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

def model(X,w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i],tf.pow(X,i))
        terms.append(term)
        print(terms)
    return tf.add_n(terms)

w = tf.Variable([0.]*num_coeffs,name = "parameters")
y_model = model(X,w)
#print(y_model)
print(y_model.shape)

cost = tf.pow(Y-y_model,2)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)     


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


for epoch in range(training_epochs):
    for(x,y) in zip(trX,trY):
        sess.run(train_op,feed_dict={X:x,Y:y})

w_val = sess.run(w)
print(w_val)

sess.close()

plt.scatter(trX,trY)
trY2 = 0

for i in range(num_coeffs):
    trY2 = w_val[i]* np.power(trX,i)
    
    
plt.plot(trX,trY2,'r')
plt.show()












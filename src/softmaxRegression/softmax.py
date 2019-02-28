#coding:utf-8
'''
Created on 2019年2月19日

@author: adp
@pdf:Machine Learning with TensorFlow MEAP v10.pdf
@page:95
'''
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import tensorflow as tf

#B
x1_label0 = np.random.normal(1,1,(100,1))
x2_label0 = np.random.normal(1,1,(100,1))
#C
x1_label1 = np.random.normal(5,1,(100,1))
x2_label1 = np.random.normal(4,1,(100,1))
#D
x1_label2 = np.random.normal(8,1,(100,1))
x2_label2 = np.random.normal(0,1,(100,1))

plt.scatter(x1_label0,x2_label0,c='r',marker='o',s=60)
plt.scatter(x1_label1,x2_label1,c='g',marker='x',s=60)
plt.scatter(x1_label2,x2_label2,c='b',marker='_',s=60)

plt.show()

xs_label0 = np.hstack((x1_label0,x2_label0))
xs_label1 = np.hstack((x1_label1,x2_label1))
xs_label2 = np.hstack((x2_label1,x2_label2))
xs = np.vstack((xs_label0,xs_label1,xs_label2))


labels = np.matrix([[1.,0.,0.]]*len(x1_label0)+[[0.,1.,0.]]*len(x1_label1)+[[0.,0.,1.]]*len(x1_label2))

arr = np.arange(xs.shape[0])
np.random.shuffle(arr)
print (arr)

xs = xs[arr,:]

label = labels[arr,:]

test_x1_label0 = np.random.normal(1,1,(10,1))
test_x2_label0 = np.random.normal(1,1,(10,1))
test_x1_label1 = np.random.normal(5,1,(10,1))
test_x2_label1 = np.random.normal(4,1,(10,1))
test_x1_label2 = np.random.normal(8,1,(10,1))
test_x2_label2 = np.random.normal(0,1,(10,1))

test_xs_label0 = np.hstack((test_x1_label0,test_x2_label0))
test_xs_label1 = np.hstack((test_x1_label1,test_x2_label1))
test_xs_label2 = np.hstack((test_x1_label2,test_x2_label2))
test_labels = np.matrix([[1.,0.,0.]]*10 + [[0.,1.,0.]]*10+[[0.,0.,1.]]*10)

print(test_xs_label0)
print(test_labels)
train_size,num_features = xs.shape


'''
@part: 4.10
@title: Using softmax regression
'''
learning_rate = 0.01
train_epochs = 1000
num_labels = 3
batch_size = 100

X = tf.placeholder(tf.float32,shape=[None,num_features])
Y = tf.placeholder(tf.float32,shape=[None,num_labels])

W = tf.Variable(tf.zeros([num_features,num_labels]))
b = tf.Variable(tf.zeros([num_labels]))
y_model = tf.nn.softmax(tf.matmul(X,W) + b)

cost = -tf.reduce_sum(Y * tf.log(y_model))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_model,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))




























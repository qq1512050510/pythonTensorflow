#coding:utf-8
'''
Created on Feb 17, 2019

@author: jyp
'''
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8
learning_rate = 0.001

#两个输入节点
x = tf.placeholder(tf.float32,shape=[None,2],name='x-input')
#回归问题一般只有一个输出节点
y_ = tf.placeholder(tf.float32,shape=[None,1],name='y-input')

#定义了一个单层的神经网络前向传播的过程，这里就是简单的加权和
#seed=None 每次随机数值不同，seed=1 每次随机数值相同
w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w1)

#定义了预测多了和少了的成本
loss_less = 10 
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*loss_more,(y_-y)*loss_less))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
print(X)
print(X.shape)
#设置回归的正确值为两个输入的和加上一个随机量。之所以要加上一个随机量是为了
#加入不可预测的噪音，否则不同损失函数的意义就不大了，因为不同损失函数都会在能完全预测正确的时候最低。
#一般来说噪音为一个均值为0的小量，所以这里的噪音设置为-0.05～0.05的随机数。
Y = [[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X]

#训练神经网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(w1.eval())
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size)%dataset_size
        end = min(start+batch_size,dataset_size)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        print("start")
        print(X[start:end])
        print("end")
        print(sess.run(w1))










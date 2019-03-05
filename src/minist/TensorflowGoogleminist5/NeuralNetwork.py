#codeing:utf-8
'''
Created on Mar 5, 2019

@author: jyp
@book:Tensorflow 实战 Google 深度学习框架
@part:第五章
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../../minist/",one_hot=True)

#MNIST数据集相关的常数。
INPUT_NODE = 784
OUTPUT_NODE = 10

#配置神经网络的参数。
LAYER1_NODE = 500
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8 #基础学习率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减度
REGULARIZATION_RATE = 0.0001 #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 3000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率

'''
 一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的向前传播结果。在这里
 定义了一个使用ReLU激活函数的三层全连接神经网络。通过加入隐含层实现了多层网络结构，
 通过ReLU激活函数实现了去线性化。在这个函数中也支持传入参数平均值的类，
 这样方便在测试时使用滑动平均模型
 '''
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
     #当没有提供滑动平均类是，直接使用参数当前的取之。
     if avg_class == None:
         #计算隐含层的前向传播结果，这里使用了ReLU激活函数
         layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
         return tf.matmul(layer1,weights2)+biases2
     else:
         layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
         return tf.matmul(layer1,avg_class.average(weights2)) + avg_class.average(biases2)
     

 
def train(mnist):
    return None
 
 




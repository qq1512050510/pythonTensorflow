#coding:utf-8
'''
Created on 2019年3月12日

@author: adp
'''
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#加载mnist_inference.py中定义的变量和前向传播的函数
import NeuralNetwork5_5_mnist_inference

#配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RAGE_DECAY = 0.99
REGULARZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

#模型保存的路径和文件名
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    #定义输入输出placeholder
    x = tf.placeholder(tf.float32,[None,NeuralNetwork5_5_mnist_inference.INPUT_NODE],name="x-input")
    y_ = tf.placeholder(tf.float32,[None,NeuralNetwork5_5_mnist_inference.OUTPUT_NODE],name="y-input")
    
    regularizer = tf.contrib.layer.l2_regularizer(REGULARZATION_RATE)
    
    #直接使用mnist_inference.py中定义的前向传播过程
    y = NeuralNetwork5_5_mnist_inference.inference(x,regularizer)
    global_step = tf.Variable(0,trainable=False)
    
    #和5.2.1小节样例中类似地定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    print(variable_averages_op)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)



#codeing:utf-8
'''
Created on Feb 28, 2019

@author: jyp
'''
import tensorflow as tf

#获取一层神经网络上的权重，并且将这个权重的L2正则化损失加入名称为‘losses’的集合中
print("你好")
def get_weight(shape,lambd):
    #生成一个变量
    var = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    #add_to_collection函数将这个新生成变量的L2正则化损失项加入集合。
    #这个函数的第一个参数‘losses’是集合的名字，第二个参数是要加入这个集合的内容
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lambd)(var))
    #返回生成的变量
    return var
x = tf.placeholder(tf.float32,shape=[None,2])
y_ = tf.placeholder(tf.float32,shape=[None,1])
batch_size = 8
#定义每一层网络中节点的个数
layer_dimension = [2,10,10,10,1]
#神经网络的层数
n_layers = len(layer_dimension)

#这个变量维护前向传播时最深层的节点，开始的时候就是输入层
cur_layer = x
#当前层的节点个数
in_dimension = layer_dimension[0]

#通过一个循环生成5层全连接的神经网络结构
for i in range(1,n_layers):
    #layer_dimension[i]为下一层的节点个数
    out_dimension = layer_dimension[i]
    print(out_dimension)
    #生成当前层中权重的变量，并将这个变量的L2正则化加入计算图上的集合
    weight = get_weight([in_dimension,out_dimension],0,001)
    






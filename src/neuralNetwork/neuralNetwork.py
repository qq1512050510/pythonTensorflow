#-*- coding:utf-8 -*-
'''
Created on Jan 29, 2019
book url: 链接:https://pan.baidu.com/s/1JtinvJ1SuAMDtfrWuW72pg  密码:g170
Page:62
@author: jyp
'''
import tensorflow as tf 
#Numpy是一个科学计算的工具包，这里通过Numpy工具生成模拟数据集
from numpy.random import RandomState
from tensorflow.python.tools.saved_model_cli import run_saved_model_with_feed_dict

#定义训练数据集batch的大小
batch_size = 8

#定义神经网络的参数，这里还是沿用 3.4.2 小节中给出的神经网络结构
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
'''
with tf.Session() as sess:
    print(sess.run(w1.initializer))
    print(sess.run(w1))
'''
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))



#在shape的一个维度上使用None可以方便使用不打的batch大小。在训练时需要把数据分成
#比较小的batch，但是在测试时，可以一次性使用全部的数据。当数据集比较小时这样比较
#方便测试，但数据集比较大时，将大量数据放入batch可能导致内存溢出

x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')

#定义神经网络前向传播的过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#定义损失函数和反向传播的算法
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
#定义规则来给出样本的标签。在这里所有x1+x2<1的样本都 被标为是正样本，（比如零件合格），
#而其他为负样本（比如零件不合格）。和TensorFlow游乐场中的表示法不大一样的地方是，
#在这里使用0来表示负样本，1来表示正样本。大部分解决分类问题的神经网络都会采用
#0和1的表示方法
Y = [[int(x1+x2<1)] for (x1,x2) in X]
print(X)
print(Y) 

#创建一个会话来运行TensorFlow程序
with tf.Session() as sess:
    #init_op = tf.initialize_all_variables()
    init_op = tf.global_variables_initializer()
    #初始化变量。
    sess.run(init_op)    
    print(sess.run(w1))
    print(sess.run(w2))
    '''
            在训练之前的神经网络参数的值：
    w1 = [[-0.81132822,1.28259876,0.06532937]
            [-2.44270396,0.0992484,0.59122431]]
    w2 = [[-0.81131822],[1.48459876],[0.06532937]]
    '''
    
    #设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        #每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size 
        end = min(start+batch_size,dataset_size)
        
        #通过选取的样本训练神经网络并更新参数。
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%1000 == 0:
            #每次选取batch_size个样本进行训练
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training steps(s),cross entropy on all data is %g"%(i,total_cross_entropy))
            '''
                                    输出结果
            Ater 0 training step(s), cross entropy on all data is 0.0674925
            Ater 1000 training step(s), cross entropy on all data is 0.0674925
            Ater 2000 training step(s), cross entropy on all data is 0.0674925
            Ater 3000 training step(s), cross entropy on all data is 0.0674925
            Ater 4000 training step(s), cross entropy on all data is 0.0674925
            
                                    通过这个结果可以发现随着训练的进行，交叉熵是逐渐变小的，交叉熵越小说明预测的结果和真是的额结果差距越小
            '''
    print(sess.run(w1))
    print(sess.run(w2))
    '''
            在训练之后神经网络的参数的值：
    w1 = [[-1.9619274,2.58235407,1.68203783]
        [-3.4681716,1.06982327,2.11788988]]
    w2 = [[-1.8247149],[2.68546653],[1.41819501]]
            可以发现这两个参数的取值已经发生了变化，这个变化就是训练的结果
            它使得这个神经网络能更好的拟合提供的训练数据
    '''
    
    
    
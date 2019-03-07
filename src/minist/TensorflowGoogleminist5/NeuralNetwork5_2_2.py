#codeing:utf-8
'''
Created on Mar 5, 2019

@author: jyp
@book:Tensorflow 实战 Google 深度学习框架
@part:第五章 2.2 节
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("../../minist/",one_hot=True)

#MNIST数据集相关的常数。
INPUT_NODE = 784
OUTPUT_NODE = 10

#配置神经网络的参数。
LAYER1_NODE = 500
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8 #基础学习率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减度
REGULARIZATION_RATE = 0.0001 #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率

'''
 一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的向前传播结果。在这里
 定义了一个使用ReLU激活函数的三层全连接神经网络。通过加入隐含层实现了多层网络结构，
 通过ReLU激活函数实现了去线性化。在这个函数中也支持传入参数平均值的类，
 这样方便在测试时使用滑动平均模型
 '''
 
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
 '''    
def inference(input_tensor,reuse=False):
    #定义第一层神经网络的变量和向前传播过程。
    with tf.variable_scope('layer1',reuse=reuse):
        '''
        根据传进来的reuse来判断是创建新变量还是使用已经创建好的。在第一次构建网络时需要
        创建新的变量，以后每次调用这个函数都直接使用reuse=True就不需要每次将变量传进来了
        '''
        weights = tf.get_variable('weights',[INPUT_NODE,LAYER1_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases',[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    
    with tf.variable_scope('layer2',reuse=reuse):
        weights = tf.get_variable('weights',[LAYER1_NODE,OUTPUT_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases',[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights) + biases        
    return layer2
        
        
 
def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name="x-input")
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name="y-input")
    
    #生成隐藏层的参数。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    #生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    #计算在当前参数下的神经网络向前传播的结果，这里给出的用于计算滑动平均的类为None,
    #所以函数不会使用参数的滑动平均值
    #y = inference(x,None,weights1,biases1,weights2,biases2)
    y = inference(x)
    '''
    定义存储训练轮数的变量。这个变量不需要甲酸滑动平均值，所以这里指定这个变量为
    不可训练的变量（trainable=False）。在使用TensorFlow训练神经网络时，
    一般会将代表训练轮数的变量指定为不可训练的参数。
    '''
    global_step = tf.Variable(0,trainable=False)
    '''
    给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。在第4章介绍过给定
    训练轮数的变量可以加快训练早起变量的更新速度
    '''
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    '''
    在所有代表神经网路参数的变量上使用滑动平均，其他辅助变量（比如global_step）就
    不需要了，tf.train_variable返回的就是图上的集合
    GraphKeys.TRAINABLE_VARIABLES中的元素。这个集合的元素是所有没有指定
    trainbale=False的参数
    '''
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    '''
    计算使用了滑动平均之后的向前传播结果。第4章中介绍过滑动平均不会改变变量本身的取值，而是会维护一个影子变量啦记录其滑动平均值。所有当需要使用这个滑动平均值时，
    需要明确调用average函数
    '''
    #average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)
    average_y = inference(x,True)
    
    #之前的调用方式
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算L2正则化损失函数
    regularization = regularizer(weights1) + regularizer(weights2)
    
    loss = cross_entropy_mean +regularization
    
    '''
    基础的学习率，随着迭代的进行，更新变量时使用的
    学习率在global_step的基础上递减
global_step 挡墙迭代的轮数
mnist.train.num_examples/BATCH_SIZE
LEARNING_RATE_DECAY

    '''
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
    
#使用tf.train.GradientDescentOptimizer 优化算法来优化算是函数。注意这里损失函数
#包含了交叉熵损失和L2正则化损失
   #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    #只优化交叉熵模型的模型优化函数的声明语句
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean, global_step=global_step)
    
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op = tf.no_op(name="train")
    
    correct_preduction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_preduction,tf.float32))
    
    #初始化会话并开始训练过程
    with tf.Session() as sess:
        #tf.initialize_all_variables().run()
        tf.global_variables_initializer().run()
        #准备验证数据。一般在神经网络的训练过程中会通过验证数据来大致判断停止的条件和评判训练的结果。
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}
        
        #迭代训练神经网络
        for i in range(TRAINING_STEPS):
            #每1000轮输出一次在数据集上的测试结果
            if i%1000 == 0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                test_acc = sess.run(accuracy,feed_dict=test_feed)
                print("After %d training step(s),validate accuracy""using average model is %g,test accuracy using average model is %g"%(i,validate_acc,test_acc))
        #产生这一轮使用的一个batch的训练数据，并运行训练过程
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
            
        #在训练结束之后，在测试数据上检测神经网络模型的最终正确率。
        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("After %d training steps(s),test accuracy using average""model is %g"%(TRAINING_STEPS,test_acc))
    
    
    
#主程序入口。
def main(argv=None):
    mnist = input_data.read_data_sets("../../minist/",one_hot=True)
    train(mnist)
    
    
if __name__=="__main__":
    tf.app.run()
                
 
 




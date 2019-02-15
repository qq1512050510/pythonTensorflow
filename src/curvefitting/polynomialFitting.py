'''
Created on 2019年2月14日

@author: jyp
@url:https://blog.csdn.net/chengshuhao1991/article/details/78545464
'''
#matplotlib inline
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

#设置plt绘图画布的尺寸
plt.rcParams["figure.figsize"] = (14,8)

#准备数据
n_observations = 100
xs = np.linspace(-3,3,n_observations)
print(type(xs))
ys = np.sin(xs) + np.random.uniform(-0.5,0.5,n_observations)

plt.scatter(xs,ys)
#plt.show()
#准备好placeholder，开好容器来装数据
X = tf.placeholder(tf.float32,name='X')
Y = tf.placeholder(tf.float32,name='Y')
#初始化参数/权重
W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')
#计算预测结果
Y_pred = tf.add(tf.multiply(X, W),b)

#添加高次项

W_2 = tf.Variable(tf.random_normal([1]),name='weight_2')
Y_pred = tf.add(tf.multiply(tf.pow(X,2),W_2),Y_pred)
W_3 = tf.Variable(tf.random_normal([1]),name='weight_3')
Y_pred = tf.add(tf.multiply(tf.pow(X, 3),W_3),Y_pred)
# W_4 = tf.Variable(tf.random_normal([1]),name = 'weight_4')
# Y_pred = tf.add(tf.multiply(tf.pow(X,4),W_4), Y_pred)

#计算损失函数

#取出xs的个数，这里是100个
sample_num = xs.shape[0]
#向量对应的点相减之后，求平方和再除以点的个数
loss = tf.reduce_sum(tf.pow(Y_pred-Y,2))/sample_num

#初始化optimizer

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#指定迭代次数，并在session里执行graph
n_samples = xs.shape[0]
init = tf.global_variables_initializer()
with tf.Session() as sess:
    #初始化所有变量
    sess.run(init)
    
    #将搜集的变量吸入事件文本，提供给tensorboard使用
    writer = tf.summary.FileWriter('./graphs',sess.graph)
    #writer = tf.train.SummaryWriter('./graphs/polynomial_reg',sess.graph)
    
    #训练模型
    for i in range(1000):
        total_loss = 0 #设定总共的损失初始值为0
        #zip:将两个列表中的对应元素分别取一个出来，形成一个元组
        for x,y in zip(xs,ys):
            _,l = sess.run([optimizer,loss],feed_dict={X:x,Y:y})
            #计算所有的损失值进行叠加
            total_loss +=l
        if i%100 == 0:
            print('Epoch{0}:{1}'.format(i,total_loss/n_samples))
    
    #关闭writer
    writer.close()
    #去除w和b的值
    W,W_2,W_3,b = sess.run([W,W_2,W_3,b])

print("W:"+str(W[0]))
print("W_2:"+str(W_2[0]))
print("W_3:"+str(W_3[0]))
#print("W_4:"+str(W_4[0]))
print("b:"+str(b[0]))

plt.plot(xs,ys,'bo',label='Real data')#真实值的散点
#预测值得拟合
plt.plot(xs,xs*W+np.power(xs,2)*W_2 + np.power(xs,3)*W_3 + b,'r',label='Predicted data')
        
#用用显示图例
plt.legend()
#显示图
plt.show()




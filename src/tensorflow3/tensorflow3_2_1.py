#-*- coding:utf-8 -*-
'''
Created on 2019年1月17日

@author: adp
'''
import tensorflow as tf

#a = tf.constant([1.0,2.0],name="a")# [1,2]做加和 提示 类型不匹配
a = tf.constant([1,2],name="a",dtype=tf.float32)
b = tf.constant([2.0,3.0],name="b")

result = a + b
print(result)
print(result.get_shape())

# 手动创建会话，手动释放会话

sess = tf.Session()
print(sess.run(result))
sess.close()


# 创建一个会话，并通过 python 中的上下文管理器 管理会话
with tf.Session() as sess:
    print(sess.run(result))
    #不需要 “sess.cose()” 来关闭会话
    # 上下文退出时关闭会话和资源释放
    
#通过设定默认会话计算张量的取值
sess = tf.Session()
with sess.as_default():
    print(result.eval())
    

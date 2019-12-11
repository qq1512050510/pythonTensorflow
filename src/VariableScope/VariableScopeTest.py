#coding:utf-8
'''
Created on 2019年3月14日

@author: adp
@book:TensorFlow 技术解析与实战
@part: 4.5变量作用域
'''
import tensorflow as tf

'''
with tf.variable_scope("foo") as foo_scope:
    v = tf.get_variable("v",[1])
    assert foo_scope.name == "foo"
with tf.variable_scope(foo_scope):
    w = tf.get_variable("w",[1])
    
with tf.variable_scope("bar"):
    with tf.variable_scope("baz") as other_scope:
        assert other_scope.name=="bar/baz"
        with tf.variable_scope(foo_scope) as foo_scope2:
            assert foo_scope2.name =="foo"
'''           

'''
with tf.variable_scope("foo",initializer=tf.constant_initializer(0.4)):
    v = tf.get_variable("v",[1])
    assert v.eval()==0.4
'''

print("test complete")


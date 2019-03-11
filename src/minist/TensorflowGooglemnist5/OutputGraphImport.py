#coding:utf-8
'''
Created on 2019年3月11日

@author: adp
'''
import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = "./model/combine_model.pb"
    
    #读取保存的模型文件，并将文件解析成对应的GraphDef Protocol Buffer。
    with gfile.FastGFile(model_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        
    '''
    将graph_def中保存的图加载到当前的图中。return_element=["add:0"]给出了返回
    的张量的名称，在保存的时候给出的是计算节点的名称，所有为“add”。在加载的时候给出的是
    张量的名称，所有是add:0
    '''
    result = tf.import_graph_def(graph_def,return_elements=["add:0"])
    print(sess.run(result))












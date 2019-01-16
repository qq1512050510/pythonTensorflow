# -*- coding:utf-8 -*-
'''
Created on 2019年1月15日

@author: adp
'''
import  tensorflow  as  tf
import  numpy  as  np
raw_data  = np.random.normal(10 ,  1 ,  100 ) #A
alpha  = tf.constant( 0.05 ) #B
curr_value  = tf.placeholder(tf.float32) #C
prev_avg  = tf.Variable( 0. ) #D
update_avg  = alpha  * curr_value  + ( 1 - alpha)  * prev_avg
init  = tf.global_variables_initializer()
with  tf.Session()  as  sess:
    sess.run(init)
    for i  in range( len (raw_data)): #E
        curr_avg  = sess.run(update_avg,  feed_dict= {curr_value: raw_data[i]})
        sess.run(tf.assign(prev_avg, curr_avg))
        print(raw_data[i], curr_avg)

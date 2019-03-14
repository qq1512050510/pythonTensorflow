#coding:utf-8
'''
Created on 2019年3月14日

@author: adp
'''
import tensorflow as tf

k = 2
max_iterations = 100
def initial_cluster_centroids(X,k):
    return X[0:k,:]

def assign_cluster(X,centroids):
    expanded_vectors = tf.expand_dims(X,0)
    expanded_centroids = tf.expand_dims(centroids,1)
    
    distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors,expanded_centroids)),2)
    mins = tf.argmin(distances,0)
    return mins

def recompute_centroids(X,Y):
    sums = tf.unsorted_segment_sum(X,Y,k)
    counts = tf.unsorted_segment_sum(tf.one_like(X),Y,k)
    return sums/counts

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #X = get_dataset(sess)
    

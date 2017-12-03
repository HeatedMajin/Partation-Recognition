# -*- coding: utf-8 -*-

import tensorflow as tf 

def Weights(shape,name):
    w = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(w,name=name)

def biases(shape,name):
    b = tf.constant(0.1,shape=shape)
    return tf.Variable(b,name=name)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


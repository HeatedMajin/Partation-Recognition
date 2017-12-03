# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 16:55:08 2017

从mnist的dev set中随便拿出一个来，跑一跑

@author: majin
"""
import tensorflow as tf 
import matplotlib.pyplot as plt
from utils import Weights,biases,conv2d,max_pool_2x2

#从mnist集中随便取一个图片测试
from tensorflow.examples.tutorials.mnist import input_data
import random


if __name__=="__main__":
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    img_index = random.randint(0,10000)
    img_1x784 = mnist.test.images[img_index:img_index+1]#有头无尾
    plt.imshow(img_1x784.reshape(28,28))
    plt.show()
    
    #input layer
    xs = tf.placeholder(tf.float32,[None,784])#28x28
    images = tf.reshape(xs,[-1,28,28,1])
    
    #将二维图片矩阵转换成多维的矩阵 1x784 -> 1x28x28x1
    images = tf.reshape(xs,[-1,28,28,1])
    
    ##conv1
    W_con1 = Weights([5,5,1,32],'W_con1')           #patch:(5,5) insize:1 outsize:32
    b_con1 = biases([32],'b_con1')                  #outsize:32
    a_conv1 = tf.nn.relu(conv2d(images,W_con1)+b_con1)           
                                                    #(28,28,32)
    a_conv1_pool = max_pool_2x2(a_conv1)            #max pool
                                                    #(14,14,32)
                                            
    ##con2
    W_con2 = Weights([5,5,32,64],'W_con2')          #patch:(5,5) insize:32 outsize:64
    b_con2 = biases([64],'b_con2')                  #outsize:64
    a_con2 = tf.nn.relu(conv2d(a_conv1_pool,W_con2)+b_con2)        
                                                    #(14,14,64)
    a_con2_pool = max_pool_2x2(a_con2)              #max pool
                                                    #(7,7,64)                              
                                       
    #plat change (7,7,64) ->  (1,7*7*64)                                     
    conv_out_plat = tf.reshape(a_con2_pool,[-1,7*7*64])
    
    
    ##fc1
    W_fc1 = Weights([7*7*64,1024],'W_fc1')          #insize:7*7*64 outsize:1024
    b_fc1 = biases([1024],'b_fc1')
    a_fc1 = tf.nn.relu(tf.matmul(conv_out_plat,W_fc1)+b_fc1)
    
    ##fc2
    W_fc2 = Weights([1024,10],'W_fc2')              #insize:7*7*64 outsize:10
    b_fc2 = biases([10],'b_fc2')
    prediction = tf.nn.softmax(tf.matmul(a_fc1,W_fc2)+b_fc2)
    
    with tf.Session() as sess:
        #拿出参数来
        saver = tf.train.Saver()
        saver.restore(sess,"saver/cnn_mnist_2000.ckpt")

        result = sess.run(prediction,feed_dict={xs:img_1x784})

        print("预测结果:"+str(sess.run(tf.arg_max(result,1))))
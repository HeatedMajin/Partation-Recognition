# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 12:31:38 2017

@author: majin
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


#####################################################
learning_rate = 1e-4
batch_size = 50
keep_probility = 0.5
train_iteration = 20000
tensorboard_path = "G://tensor_logs/cnn_mnist_"+str(train_iteration)
ckpt_path  = "saver/cnn_mnist_"+str(train_iteration)+".ckpt"
#####################################################


def calc_accuracy(images,labels):
    '''
        计算准确率
    '''
    global prediction
    with tf.name_scope("calc_accuracy"):
        y_pre = sess.run(prediction,feed_dict={xs:images,keep_prob:1})    #shape = m,10
        correct_pre = tf.equal(tf.argmax(y_pre,1),tf.argmax(labels,1))     #shape = m,1

        accuracy = tf.reduce_mean(tf.cast(correct_pre,"float"))
        result = sess.run(accuracy,feed_dict={xs:images,ys:labels,keep_prob : 1})
    return result

def Weights(shape,name):
    w = tf.truncated_normal( shape,stddev=0.1)
    return tf.Variable(w,name=name)

def biases(shape,name):
    b = tf.constant(0.1,shape=shape)
    return tf.Variable(b,name=name)

def conv2d(x,W,name=None):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME",name=name)

def max_pool_2x2(x,name=None):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name=name)
    
#定义输入,输出
with tf.name_scope("inputs"):  
    xs = tf.placeholder(tf.float32,[None,784],name="x_input")#28x28
    ys = tf.placeholder(tf.float32,[None,10],name="y_input")
    keep_prob = tf.placeholder(tf.float32,name="keep_prob")

#input layer
#将原先的图片矩阵转换成多维的数据 #5000x784 -> 5000x28x28x1
with tf.name_scope("matric_to_multi_vector"):  
    images = tf.reshape(xs,[-1,28,28,1])


##conv1
with tf.name_scope("Convolution1"):
    W_con1 = Weights([5,5,1,32],'W_con1')           #patch:(5,5) insize:1 outsize:32
    tf.summary.histogram("W_con1",W_con1)    
    
    b_con1 = biases([32],'b_con1')                  #outsize:32
    tf.summary.histogram("b_con1",b_con1)
    
    a_conv1 = tf.nn.relu(conv2d(images,W_con1)+b_con1,name="a_con1")           
                                                    #(28,28,32)
    a_conv1_pool = max_pool_2x2(a_conv1,name="a_conv1_maxpool")#max pool
                                                    #(14,14,32)

##con2
with tf.name_scope("Convolution2"):
    W_con2 = Weights([5,5,32,64],'W_con2')          #patch:(5,5) insize:32 outsize:64
    tf.summary.histogram("W_con2",W_con2)
    
    b_con2 = biases([64],'b_con2')                  #outsize:64
    tf.summary.histogram("b_con2",b_con2)
    
    a_con2 = tf.nn.relu(conv2d(a_conv1_pool,W_con2)+b_con2,name="a_con2")        
                                                    #(14,14,64)
    a_con2_pool = max_pool_2x2(a_con2,name="a_conv2_maxpool")#max pool
                                                    #(7,7,64)
                                                                                
#plat change (7,7,64) ->  (1,7*7*64)        
with tf.name_scope("multi_matric_to_vacter"):                             
    conv_out_plat = tf.reshape(a_con2_pool,[-1,7*7*64])


##fc1
with tf.name_scope("full_connected_layer1"):
    W_fc1 = Weights([7*7*64,1024],'W_fc1')          #insize:7*7*64 outsize:1024
    tf.summary.histogram("W_fc1",W_fc1)
    
    b_fc1 = biases([1024],'b_fc1')
    tf.summary.histogram("b_fc1",b_fc1)
    
    a_fc1 = tf.nn.relu(tf.matmul(conv_out_plat,W_fc1)+b_fc1,name="fc1_a")    
    a_fc1_dropout = tf.nn.dropout(a_fc1,keep_prob=keep_prob,name="fc1_dropout")

##fc2
with tf.name_scope("full_connected_layer_2"):
    W_fc2 = Weights([1024,10],'W_fc2')              #insize:7*7*64 outsize:10
    tf.summary.histogram("W_fc2",W_fc2)
    
    b_fc2 = biases([10],'b_fc2')
    tf.summary.histogram("b_fc2",b_fc2)
    
    prediction = tf.nn.softmax(tf.matmul(a_fc1_dropout,W_fc2)+b_fc2,name="fc2_a")
    


#定义损失函数(交叉熵)
with tf.name_scope("loss_calc"):  
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))
    tf.summary.scalar("loss",cross_entropy)
    
#定义train过程
with tf.name_scope("train"):  
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#开始迭代训练
sess = tf.Session()
sess.run(tf.initialize_all_variables())

#tensorboard draw
train_write = tf.summary.FileWriter(tensorboard_path+"/train",sess.graph)
test_write = tf.summary.FileWriter(tensorboard_path+"/test",sess.graph)
merge_op = tf.summary.merge_all()

test_xs = mnist.test.images[:1024]
test_ys = mnist.test.labels[:1024]
for i in range(train_iteration):
    #取batch
    batch_xs,batch_ys = mnist.train.next_batch(batch_size)
    
    #训练batch
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:keep_probility})
    #每100步计算准确率
    if i%100 ==0:        
        print(calc_accuracy(test_xs,test_ys))
        
        train_result = sess.run(merge_op,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:1})
        train_write.add_summary(train_result,i)
        test_result = sess.run(merge_op,feed_dict={xs:test_xs,ys:test_ys,keep_prob:1})
        test_write.add_summary(test_result,i)

#将训练的参数保存起来
saver =  tf.train.Saver({'W_con1':W_con1,'b_con1':b_con1,
                         'W_con2':W_con2,'b_con2':b_con2,
                         'W_fc1':W_fc1,'b_fc1':b_fc1,
                         'W_fc2':W_fc2,'b_fc2':b_fc2})
save_path = saver.save(sess,ckpt_path)
print("Save \"W\"s and \"b\"s at:"+save_path)

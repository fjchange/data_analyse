import numpy as np
import tensorflow as tf

#给所有的权重赋值并且创建
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.Constant(shape,0.1)
    return tf.Variable(initial)

x=tf.placeholder(tf.float32,[None,224,224,3])
y=tf.placeholder(tf.float32,[None,1000])
keep_prob=tf.placeholder(tf.float32)
x_image=tf.div(x,255.)#归一化

#--------------first part-----------------
conv1_1=tf.layers.conv2d(inputs=x_image,filters=64,kernel_size=[3,3],padding='same',use_bias=True,activation=tf.nn.relu)
conv1_2=tf.layers.conv2d(inputs=conv1_1,filters=64,kernel_size=[3,3],padding='same',use_bias=True,activation=tf.nn.relu)

pool1=tf.layers.max_pooling2d(inputs=conv1_2,pool_size=[2,2],strides=2,padding='valid')

#----------second part--------------------
conv2_1=tf.layers.conv2d(inputs=pool1,filters=128,kernel_size=[3,3],padding='same',use_bias=True,activation=tf.nn.relu)
conv2_2=tf.layers.conv2d(inputs=conv2_1,filters=128,kernel_size=[3,3],padding='same',use_bias=True,activation=tf.nn.relu)

pool2=tf.layers.max_pooling2d(inputs=conv2_2,pool_size=[2,2],strides=2,padding='valid')

#---------third part----------------------
conv3_1=tf.layers.conv2d(inputs=pool2,filters=256,kernel_size=[3,3],padding='same',use_bias=True,activation=tf.nn.relu)
conv3_2=tf.layers.conv2d(inputs=conv3_1,filters=256,kernel_size=[3,3],padding='same',use_bias=True,activation=tf.nn.relu)
conv3_3=tf.layers.conv2d(inputs=conv3_2,filters=256,kernel_size=[3,3],padding='same',use_bias=True,activation=tf.nn.relu)

pool3=tf.layers.max_pooling2d(inputs=conv3_3,pool_size=[2,2],strides=2,padding='valid')

#--------fourth part---------------------
conv4_1=tf.layers.conv2d(inputs=pool3,filters=512,kernel_size=[3,3],padding='same',use_bias=True,activation=tf.nn.relu)
conv4_2=tf.layers.conv2d(inputs=conv4_1,filters=512,kernel_size=[3,3],padding='same',use_bias=True,activation=tf.nn.relu)
conv4_3=tf.layers.conv2d(inputs=conv4_2,filters=512,kernel_size=[3,3],padding='same',use_bias=True,activation=tf.nn.relu)

pool4=tf.layers.max_pooling2d(inputs=conv4_3,pool_size=[2,2],strides=2,padding='valid')

#--------fifth part----------------------
conv5_1=tf.layers.conv2d(inputs=pool4,filters=512,kernel_size=[3,3],padding='same',use_bias=True,activation=tf.nn.relu)
conv5_2=tf.layers.conv2d(inputs=conv5_1,filters=512,kernel_size=[3,3],padding='same',use_bias=True,activation=tf.nn.relu)
conv5_3=tf.layers.conv2d(inputs=conv5_2,filters=512,kernel_size=[3,3],padding='same',use_bias=True,activation=tf.nn.relu)

pool5=tf.layers.max_pooling2d(inputs=conv5_3,pool_size=[2,2],strides=2,padding='valid')

#--------sixth part---------------------
fc6_1=tf.contrib.layers.fully_connected(inputs=pool5,num_outputs=4096)
dropout_1=tf.nn.dropout(fc6_1,keep_prob=keep_prob)
fc6_2=tf.contrib.layers.fully_connected(inputs=dropout_1,num_outputs=4096)
dropout_2=tf.nn.dropout(fc6_2,keep_prob=keep_prob)
fc6_3=tf.contrib.layers.fully_connected(inputs=dropout_2,num_outputs=1000)
output=tf.nn.softmax(fc6_3)



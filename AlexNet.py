import tensorflow as tf

x=tf.placeholder(tf.float32,[None,224,224,3])
y=tf.placeholder(tf.float32,[None,1000])
x_image=tf.div(x,255.)

#---------first part----------------------
conv1_1=tf.layers.conv2d(inputs=x_image,filters=48,kernel_size=[11,11],)
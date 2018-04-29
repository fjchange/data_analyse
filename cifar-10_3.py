import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#获得mnist文件的路径
#mnist=input_data.read_data_sets('H:\\Users\\kiwi feng\\Desktop\\',one_hot=True)

train_sample_epoch=50000
test_sample_epoch=10000
data_dir='.\\cifar-10-batches-py'
batch_size=100

#现在我们获得一个字典
def uppickle(file):
    import pickle
    with open(file,'rb')as fo:
        dict=pickle.load(fo,encoding='bytes')
    return dict


sess = tf.InteractiveSession()


#产生一个权重矩阵，shape 为四维的【batch,data_height,data_width,channels]
def weight_variable(shape,name):
    #stddev为标准差
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name=name)
#产生一个shape大小的偏置
def bias_variable(shape,name):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)


#input 的图片尺寸为列数为3072的矩阵
#输入的labels是10个
#转换为原有尺寸大小
x=tf.placeholder(tf.float32,[None,3072])
y_=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,32,32,3])
x_image=tf.div(x_image,255.)

conv1_1=tf.layers.conv2d(inputs=x_image,
                       filters=64,
                       kernel_size=[3,3],
                       padding='same',
                       use_bias=True,
                       activation=tf.nn.relu
                       )
#conv2 size out_channels=64
conv1_2=tf.layers.conv2d(inputs=conv1_1,
                       filters=64,
                       kernel_size=[3,3],
                       padding='same',
                       use_bias=True,
                       activation=tf.nn.relu
                       )
conv1_3=tf.layers.conv2d(inputs=conv1_2,
                       filters=32,
                       kernel_size=[1,1],
                       padding='same',
                       use_bias=True,
                       activation=tf.nn.relu
                       )
#conv1_2_norm=tf.nn.local_response_normalization(conv1_2)
#max_pooling 下降到一半 16x16
pool1=tf.layers.max_pooling2d(inputs=conv1_3,
                              pool_size=[2,2],
                              strides=2,
                              padding='valid')

#
conv2_1=tf.layers.conv2d(inputs=pool1,
                       filters=128,
                       kernel_size=[3,3],
                       padding='same',
                       use_bias=True,
                       activation=tf.nn.relu
                       )
conv2_2=tf.layers.conv2d(inputs=conv2_1,
                       filters=128,
                       kernel_size=[3,3],
                       padding='same',
                       use_bias=True,
                       activation=tf.nn.relu
                       )
conv2_3=tf.layers.conv2d(inputs=conv2_2,
                       filters=128,
                       kernel_size=[3,3],
                       padding='same',
                       use_bias=True,
                       activation=tf.nn.relu
                       )
conv2_4=tf.layers.conv2d(inputs=conv2_3,
                       filters=64,
                       kernel_size=[1,1],
                       padding='same',
                       use_bias=True,
                       activation=tf.nn.relu
                       )
#下降到8*8
#conv2_3_norm=tf.nn.local_response_normalization(conv2_3)

pool2=tf.layers.max_pooling2d(
    inputs=conv2_4,
    pool_size=[2,2],
    strides=2,
    padding='valid'
)

#pool_3=tf.nn.local_response_normalization(pool2)

W_fc1=weight_variable([8*8*64,1024],'W_fc1')
b_fc1=bias_variable([1024],'b_fc1')
h_pool2_flat=tf.reshape(pool2,[-1,8*8*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

W_fc2=weight_variable([1024,128],'W_fc2')
b_fc2=bias_variable([128],'b_fc2')
h_fc2=tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
h_fc2_drop=tf.nn.dropout(h_fc2,keep_prob)

W_fc3=weight_variable([128,10],'W_fc3')
b_fc3=bias_variable([10],'b_fc3')

y_conv=tf.nn.softmax(tf.matmul(h_fc2_drop,W_fc3)+b_fc3)


global_step = tf.Variable(0, trainable=False)
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
learning_rate=tf.train.exponential_decay(learning_rate=1e-4,global_step=global_step,decay_steps=1000,decay_rate=0.96,staircase=True)
train_step=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy,global_step=global_step)

correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.global_variables_initializer().run()
#saver=tf.train.Saver({'W_conv1':W_conv1,'b_conv1':b_conv1,'W_conv2':W_conv2,'b_conv2':b_conv2,'W_fc1':W_fc1,"b_fc1":b_fc1,'W_fc2':W_fc2,"b_fc2":b_fc2})

test=uppickle(data_dir+'\\test_batch')
test_list=np.array(test[b'data'])
test_np=np.resize(test_list,[10000,3072])
labels = np.array(test[b'labels'])
test_labels=(np.arange(10)==labels[:,None]).astype(np.float32)
test_size=100

epoch_time=60
for c in range(epoch_time):
    print(learning_rate.eval(session=sess))
    for i in range(5):

        batch=uppickle(data_dir+'\\data_batch_'+(i+1).__str__())
        batch_list=np.array(batch[b'data'])
        batch_np=np.resize(batch_list,[10000,3072])
        labels = np.array(batch[b'labels'])
        y_labels=(np.arange(10)==labels[:,None]).astype(np.float32)


        for t in range(100):
            if t%100==0:
                train_accuracy=accuracy.eval(feed_dict={x:batch_np[t*batch_size:(t+1)*batch_size],y_:y_labels[t*batch_size:(t+1)*batch_size],keep_prob:1.0})
                print("epoch %d step %d,trainning accuracy %g"%(c,t+100*i,train_accuracy))
            train_step.run(feed_dict={x:batch_np[t*batch_size:(t+1)*batch_size],y_:y_labels[t*batch_size:(t+1)*batch_size],keep_prob:0.5})
    test_accuracy=accuracy.eval(feed_dict={x:test_np[:1000],y_:test_labels[:1000],keep_prob:1.0})
    print("epoch %d test accuracy %g" % (c, test_accuracy))

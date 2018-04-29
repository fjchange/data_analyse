import tensorflow as tf
import LeNet

model=tf.train.import_meta_graph('Model/model.ckpt.meta')
sess=tf.InteractiveSession()
model.restore(sess,'Model/model.ckpt')
print(sess.run(tf.get_default_graph().get_Tensor))
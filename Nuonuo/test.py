import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.Variable(tf.random_normal([3,4],stddev=0.1),name='a')
b = tf.constant([1])
c = tf.constant([1,1,2],dtype=tf.float32)
d = tf.Variable(tf.zeros(shape=[5],dtype=tf.float32),name='d')
y_most =tf.Variable(tf.zeros(shape=tf.shape(c),dtype=tf.float32))

for i in range(d.get_shape()[0]):
    print(i)

# with tf.Session(config=tf.ConfigProto(log_device_placement = True)) as sess:
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # print(sess.run(tf.shape(c)))

    print(sess.run(y_most))

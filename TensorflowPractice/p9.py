# from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
# mnist=input_data.read_data_sets("/home/tolic/dataSet/MNIST_data",one_hot=True)
b=tf.Variable(tf.random_normal(shape=[2,4],seed=1))
a=tf.constant(0.1,dtype=tf.float32,shape=[4])
c=tf.add(b,a)
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(b))
    print(sess.run(a))
    print(sess.run(c))
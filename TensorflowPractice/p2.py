import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/tolic/dataSet/MNIST_data",
                                  one_hot=True)
with tf.Session() as sess:
    x_test = mnist.test.images
    x = tf.reshape(x_test, [10000, 28,28, 1])
    xx = sess.run(x)
    print(x)
    # print(xx)

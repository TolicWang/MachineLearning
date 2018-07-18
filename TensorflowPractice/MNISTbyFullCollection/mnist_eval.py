import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
def evaluate(mnist):
    x = tf.placeholder(
        tf.float32,[None,mnist_inference.INPUT_NODE],name='x-input')
    y_ = tf.placeholder(
        tf.float32,[None,mnist_inference.OUTPUT_NODE],name='y-input')
    y = mnist_inference.inference(x,None)
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './model/model.ckpt')
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('Test accuracy:',test_acc)
def main(argv =None):
    mnist = input_data.read_data_sets("/home/tolic/dataSet/MNIST_data",
                                      one_hot=True)
    evaluate(mnist)
if __name__ == '__main__':
    tf.app.run()
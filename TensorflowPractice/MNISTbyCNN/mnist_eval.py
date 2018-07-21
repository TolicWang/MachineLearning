import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

BATCH_SIZE = 10
def evaluate(mnist):
    x = tf.placeholder(
        tf.float32,[BATCH_SIZE,
                    mnist_inference.IMAGE_SIZE,
                    mnist_inference.IMAGE_SIZE,
                    mnist_inference.NUM_CHANNELS],name='x-input')
    y_ = tf.placeholder(
        tf.float32,[BATCH_SIZE,mnist_inference.OUTPUT_NODE],name='y-input')
    y = mnist_inference.inference(x,False,None)
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, './model/model.ckpt')
        x_test = mnist.test.images[:BATCH_SIZE]
        reshaped_x_test = tf.reshape(x_test, [BATCH_SIZE, mnist_inference.IMAGE_SIZE,
                                mnist_inference.IMAGE_SIZE,
                                mnist_inference.NUM_CHANNELS])
        xx=sess.run(reshaped_x_test)
        test_acc = sess.run(accuracy, feed_dict={x:xx, y_: mnist.test.labels[:BATCH_SIZE]})
        print('Test accuracy:',test_acc)
def main(argv =None):
    mnist = input_data.read_data_sets("/home/tolic/dataSet/MNIST_data",
                                      one_hot=True)
    evaluate(mnist)
if __name__ == '__main__':
    tf.app.run()
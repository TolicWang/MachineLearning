import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import inference
import train
def evaluate():
    x = tf.placeholder(
        tf.float32,[None,inference.INPUT_NODE],name='x-input')
    y_ = tf.placeholder(
        tf.int64,[None],name='y-input')
    y = inference.inference(x,None)
    correct_prediction=tf.equal(tf.argmax(y,1),y_)
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './model/model.ckpt')
        _,x_test,_,y_test=train.Load_data()
        test_feed = {x: x_test.toarray(), y_: y_test}
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('Test accuracy:',test_acc)
def main(argv =None):

    evaluate()
if __name__ == '__main__':
    tf.app.run()
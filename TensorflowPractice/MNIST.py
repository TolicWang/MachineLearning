import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/home/tolic/dataSet/MNIST_data",one_hot=True)
INPUT_NODE=784
OUTPUT_NODE=10
LAER1_NODE=500
BATCH_SIZE=100
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRAINING_STEPS=300

def inference(input_tensor,weights1,biases1,weights2,biases2):
    layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
    return tf.matmul(layer1,weights2)+biases2

def train(mnist):
    x=tf.placeholder(dtype=tf.float32,shape=[None,INPUT_NODE],name='x-input')
    y_=tf.placeholder(dtype=tf.float32,shape=[None,OUTPUT_NODE],name='y-input')
    weights1=tf.Variable(tf.truncated_normal(shape=[INPUT_NODE,LAER1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAER1_NODE]))
    weights2=tf.Variable(tf.truncated_normal(shape=[LAER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    y=inference(x,weights1,biases1,weights2,biases2)
    global_step=tf.Variable(0,trainable=False)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization=regularizer(weights1)+regularizer(weights2)
    loss=cross_entropy_mean+regularization
    learning_rate=tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE,
                                             global_step=global_step,
                                             decay_steps=mnist.train.num_examples/BATCH_SIZE,
                                             decay_rate=LEARNING_RATE_DECAY)
    train_step=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(
        loss=loss,global_step=global_step)

    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        test_feed={x:mnist.test.images,y_:mnist.test.labels}
        saver=tf.train.Saver()
        for i in range(TRAINING_STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step,feed_dict={x:xs,y_:ys})
            if i % 100 == 0:
                validate_acc,l=sess.run([accuracy,loss],feed_dict=validate_feed)
                print('After %d training step(s), validation accuracy is'
                      ' %g, loss is %s'%(i,validate_acc,l))
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print('After %d training step(s), test accuracy is %g'%(TRAINING_STEPS,test_acc))
        saver.save(sess,'./model.ckpt')

if __name__ == '__main__':
    train(mnist)




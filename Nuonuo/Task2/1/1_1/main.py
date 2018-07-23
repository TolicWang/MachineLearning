import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/tolic/dataSet/MNIST_data",
                                  one_hot=True)

LEARNING_RATE = 0.3
BATCH_SIZE = 100
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 2500

x = tf.placeholder(tf.float32,shape=[None,784],name='x-input')
y_ = tf.placeholder(tf.float32,shape=[None,10],name='y-input')

with tf.variable_scope('p'):
    weight = tf.get_variable(name='weight',shape=[784,10],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
    biase = tf.get_variable(name='biase',shape=[10],
                            initializer=tf.constant_initializer(0.1))
y = tf.nn.softmax(tf.matmul(x,weight)+biase)
loss= -tf.reduce_mean(y_*tf.log(y))

regularization = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)(weight)
losses = loss + regularization
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss=losses)


with tf.Session() as sess:
    print("start training...\n")
    tf.global_variables_initializer().run()
    for i in range(TRAINING_STEPS):
        batch_x,batch_y = mnist.train.next_batch(BATCH_SIZE)
        sess.run(train_step,feed_dict={x:batch_x,y_:batch_y})
        if i % 100 == 0:
            print("loss:",sess.run(losses,feed_dict={x:batch_x,y_:batch_y}))

    corrcet_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(corrcet_prediction,tf.float32))
    print("Acc:",sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist=input_data.read_data_sets("./dataSet/MNIST_data",one_hot=True)
DATA_SIZE=mnist.train.num_examples
INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500
BATCH_SIZE=100
TRAINING_STEPS=5000

W1=tf.Variable(tf.random_normal([INPUT_NODE,LAYER1_NODE],stddev=1,seed=1))
W2=tf.Variable(tf.random_normal([LAYER1_NODE,OUTPUT_NODE],stddev=1,seed=1))
b1=tf.Variable(tf.random_normal([1,LAYER1_NODE],stddev=1,seed=1))
b2=tf.Variable(tf.random_normal([1,OUTPUT_NODE],stddev=1,seed=1))

x=tf.placeholder(dtype=tf.float32,shape=(None,INPUT_NODE),name='x-input')
y_=tf.placeholder(dtype=tf.float32,shape=(None,OUTPUT_NODE),name='y-output')

a=tf.nn.relu(tf.matmul(x,W1)+b1)
y=tf.nn.relu(tf.matmul(a,W2)+b2)

y=tf.nn.softmax(y)
cross_entropy=tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
regularizer=tf.contrib.layers.l2_regularizer(0.02)
regularization=regularizer(W1)+regularizer(W2)
loss=cross_entropy+regularization

train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train_step=tf.train.AdamOptimizer(0.0005).minimize(cross_entropy)

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    validation_feed={x:mnist.validation.images,y_:mnist.validation.labels}
    for i in range(TRAINING_STEPS):
        start=(i*BATCH_SIZE)%DATA_SIZE
        end=min(start+BATCH_SIZE,DATA_SIZE)
        sess.run(train_step,feed_dict={x:mnist.train.images[start:end],
                                        y_:mnist.train.labels[start:end]})
        if i%500==0:
            total_cross_entropy=sess.run(cross_entropy,feed_dict=validation_feed)
            acc=sess.run(accuracy,feed_dict=validation_feed)
            print('loss:%d,acc:%d'%(total_cross_entropy,acc))

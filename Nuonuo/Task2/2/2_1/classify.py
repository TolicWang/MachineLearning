import tensorflow as tf
from sentence2word import Load_train_and_test_data
from datetime import datetime
INPUT_NODE = 1024
OUTPUT_NODE = 3510
LEARNING_RATE = 0.5
BATCH_SIZE = 300
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 50000
DATA_SIZE = 738793

x_train,y_train,x_test,y_test = Load_train_and_test_data()

x = tf.placeholder(tf.float32,shape=[None,INPUT_NODE],name='x-input')
y_ = tf.placeholder(tf.int64,shape=[None],name='y-input')
with tf.variable_scope('p'):
    weight = tf.get_variable(name='weight',shape=[INPUT_NODE,OUTPUT_NODE],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
    biase = tf.get_variable(name='biase',shape=[OUTPUT_NODE],
                            initializer=tf.constant_initializer(0.1))
logit = tf.matmul(x,weight)+biase
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit,labels=y_)
cross_entropy_mean = tf.reduce_mean(cross_entropy)

regularization = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)(weight)
losses = cross_entropy_mean + regularization
with tf.device('/gpu:0'):
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss=losses)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print("Start training...\n")
    tf.global_variables_initializer().run()
    last = datetime.now()
    for i in range(TRAINING_STEPS):
        start = (i * BATCH_SIZE) % DATA_SIZE
        end = min(start + BATCH_SIZE, DATA_SIZE)
        batch_x = x_train[start:end]
        batch_y = y_train[start:end]
        sess.run(train_step,feed_dict={x:batch_x,y_:batch_y})
        if i % 100 == 0:
            now = datetime.now()
            print(i,"loss:",
                  sess.run(losses,feed_dict={x:batch_x,y_:batch_y}),'times ',now-last)
            last = now
    corrcet_prediction = tf.equal(tf.argmax(logit,1),y_)
    accuracy = tf.reduce_mean(tf.cast(corrcet_prediction,tf.float32))
    print("Acc:",sess.run(accuracy,feed_dict={x:x_test,y_:y_test}))

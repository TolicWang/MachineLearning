import tensorflow as tf
from numpy.random import RandomState
import numpy as np
batch_size=32

w1=tf.Variable(tf.random_normal([3,2],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))

x=tf.placeholder(dtype=tf.float32,shape=(None,3),name='x-input')
y_=tf.placeholder(dtype=tf.float32,shape=(None, 3),name='y-input')

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

rdm=RandomState(13)
dataset_size=512
X=rdm.rand(dataset_size,3)
Y=np.reshape([[int(x1+x2+x3<1)]for (x1,x2,x3)in X],dataset_size)

eye = np.eye(3)
Y = eye[Y,:]

# y=tf.nn.softmax(y)
# cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
cross_entropy=tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y))

train_step=tf.train.AdamOptimizer(0.002).minimize(cross_entropy)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)

    # print(sess.run(w2))
    STEPS=5000
    for i in range(STEPS):
        start=(i*batch_size)%dataset_size
        end=min(start+batch_size,dataset_size)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        # sess.run(train_step,feed_dict={x:X,y_:Y})

        if i%1000==0:
            total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            # print("After %d training setp(s),cross entropy on all data is %g"
            #       %(i,total_cross_entropy))
            print(total_cross_entropy)


    print(x)
    # print(sess.run(w1))
    # print(sess.run(w2))
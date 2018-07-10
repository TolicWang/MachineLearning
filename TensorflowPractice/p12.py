import tensorflow as tf
from random import randint

dims = 8
pos = [5,4]

logits=tf.random_normal(shape=[2,8],seed=1)
labels = tf.constant([[0,0,0,0,0,1,0,0],[0,0,0,0,1,0,0,0]])

res1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
res2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(labels,1))

with tf.Session() as sess:
    a, b = sess.run([res1, res2])
    print (a, b)
    # print(sess.run(labels))
    # print(sess.run(logits))

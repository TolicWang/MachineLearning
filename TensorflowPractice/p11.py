import tensorflow as tf
y=tf.constant([[0,0,0,1,0],[0,0,1,0,0]],dtype=tf.float32)
logits=tf.random_normal(shape=[2,5],seed=1)
# logits=tf.Variable(tf.random_normal(shape=[2,5]),dtype=tf.float32)

loss1=tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y)
loss1=tf.reduce_mean(loss1)
loss2=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=tf.argmax(y,1))
loss2=tf.reduce_mean(loss2)
with tf.Session() as sess:
    a=sess.run(loss1)
    b=sess.run(loss2)
    print(a, b)

    sess.run(tf.global_variables_initializer())
    a,b=sess.run([loss1,loss2])
    print(a,b)



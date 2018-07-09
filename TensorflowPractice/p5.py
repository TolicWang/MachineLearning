import tensorflow as tf
v=tf.constant([[1,2,3],[4,5,6]],dtype=tf.float32)
p=tf.constant([1,2.71,4],dtype=tf.float32)
with tf.Session() as sess:
    # init_op=tf.global_variables_initializer()
    # sess.run(init_op)
    # print(sess.run(v))
    # print(sess.run(tf.clip_by_value(v,2,5)))
    print(sess.run(tf.log(p)))
    print(tf.log(p).eval())

import tensorflow as tf

with tf.variable_scope('v1',reuse=False):
    a=tf.get_variable('a',shape=[3,4],initializer=tf.zeros_initializer())
with tf.variable_scope('v2',reuse=False):
    a=tf.get_variable('a',shape=[4,4],initializer=tf.zeros_initializer())


with tf.variable_scope('v1',reuse=True):
    a=tf.get_variable('a')
with tf.variable_scope('v2',reuse=True):
    b=tf.get_variable('a')


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(a))
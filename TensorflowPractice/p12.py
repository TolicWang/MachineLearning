import tensorflow as tf

def f1():
    v1=tf.Variable(tf.constant(1.0,shape=[1],name='v1'))
    v2=tf.Variable(tf.constant(2.0,shape=[1],name='v2'))
    result = (v1+v2)*5
    # y=2*result
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # sess.run(y)
        saver.save(sess,'./model.ckpt')
def f2():
    v1=tf.Variable(tf.constant(12.0,shape=[1],name='v1'))
    v2=tf.Variable(tf.constant(1112.0,shape=[1],name='v2'))
    # result = tf.Variable(tf.constant(12.0,shape=[1],name='v3'))
    r= (v1+v2)*5
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess,'./model.ckpt')
        print(sess.run(r))


f2()
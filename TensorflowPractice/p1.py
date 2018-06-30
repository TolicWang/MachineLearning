import tensorflow as tf
import numpy as np
A=tf.truncated_normal([2,3])
B=tf.fill([2,3],5.0)
sess=tf.Session()
print(sess.run(A))
print(sess.run(B))
print(sess.run(A+B))

C=np.random.rand(3,6)
print(C)
D=tf.convert_to_tensor(C)
print(sess.run(D))

print(sess.run(tf.nn.relu([-3,3,10])))
print(sess.run(tf.nn.relu6([-3,3,10])))
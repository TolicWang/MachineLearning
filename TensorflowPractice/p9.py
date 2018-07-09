from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist=input_data.read_data_sets("./dataSet/MNIST_data",one_hot=True)

print(mnist.train.labels[0])
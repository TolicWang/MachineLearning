import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/home/tolic/dataSet/MNIST_data",one_hot=True)
INPUT_NODE=784
OUTPUT_NODE=10
LAER1_NODE=500
BATCH_SIZE=100
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRAINING_STEPS=30000

def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    if avg_class == None:
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2)+biases2
    else:
        return None
def train(mnist):
    x=tf.placeholder(dtype=tf.float32,shape=[None,INPUT_NODE],name='x-input')
    y_=tf.placeholder(dtype=tf.float32,shape=[None,OUTPUT_NODE],name='y-output')
    weights1=tf.Variable(tf.truncated_normal(shape=[INPUT_NODE,LAER1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAER1_NODE]))
    weights2=tf.Variable(tf.truncated_normal(shape=[LAER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    y=inference(x,None,weights1,biases1,weights2,biases2)

    global_step=tf.Variable(0,trainable=False)

    cross_entropy=tf.nn.s
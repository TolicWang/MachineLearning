import pickle
import os
import tensorflow as tf
import inference
def Load_Traindata_Testdata_with_Tfidf():
    p = open('./data/train_and_test_data', 'rb')
    data = pickle.load(p)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    print('Load TF-IDF succesfully! shape:',x_train.shape)
    return x_train,x_test,y_train,y_test
def Load_data():
    x_train, x_test, y_train, y_test = Load_Traindata_Testdata_with_Tfidf()
    p = open('./data/indices', 'rb')
    data = pickle.load(p)
    indices = data['indices']
    most_importance_feature = indices[:1000]
    x_train = x_train[:,most_importance_feature]
    x_test = x_test[:,most_importance_feature]
    print('Load training data succesfully! shape:', x_train.shape)
    return x_train,x_test,y_train,y_test
BATCH_SIZE = 1000
DATA_SIZE = 738793
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 3000
MODEL_SAVE_PATH = './model'
MODEL_NAME = 'model.ckpt'

def train(X,Y):
    x = tf.placeholder(
        tf.float32,[None,inference.INPUT_NODE],name='x-input')
    y_ = tf.placeholder(
        tf.int32,[None],name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = inference.inference(x,regularizer)
    global_step = tf.Variable(0,trainable=False)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,
                                                                   labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               DATA_SIZE/BATCH_SIZE,
                                               LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss=loss,global_step=global_step)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            start = (i*BATCH_SIZE)%DATA_SIZE
            end = min(start + BATCH_SIZE,DATA_SIZE)
            # print(i,start,end)
            _,loss_value,step = sess.run([train_step,loss,global_step],
                                         feed_dict={x:X[start:end].toarray(),y_:Y[start:end]})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is "
                      "%g."%(step,loss_value))
        saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME))

def main(argv =None):
    x_train, x_test, y_train, y_test = Load_data()
    train(x_train,y_train)
if __name__ == '__main__':
    tf.app.run()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels
x_valid = mnist.validation.images
y_valid = mnist.validation.labels
x_test = mnist.test.images
y_test = mnist.test.labels

num_input = 28
timesteps = 28
n_classes = 10

learning_rate = 0.01
epochs = 10
batch_size = 100
display_freq = 100

num_hidden_units = 50

x = tf.placeholder(tf.float32, shape=[None,timesteps,num_input],name='X')
y = tf.placeholder(tf.float32, shape=[None,n_classes],name='Y')

initer = tf.truncated_normal_initializer(stddev=0.01)
W = tf.get_variable('W',
                    dtype=tf.float32,
                    shape=(num_hidden_units, n_classes),
                    initializer=initer)

initial = tf.constant(0, shape=(n_classes,), dtype=tf.float32)
b = tf.get_variable('b',
                    dtype=tf.float32,
                    initializer=initial)

def randomize(x,y):
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x,shuffled_y


def get_next_batch(x,y,start,end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch

def RNN(x, weights, biases, timesteps, num_hidden):
    x = tf.unstack(x, timesteps, 1)
    rnn_cell = rnn.BasicRNNCell(num_hidden)
    states_series, current_state = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(current_state, weights) + biases

output_logits = RNN(x,W,b,timesteps,num_hidden_units)
y_pred = tf.nn.softmax(output_logits)

cls_prediction = tf.argmax(output_logits, axis=1, name='prediction')

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits),name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
correct_prediction = tf.equal(tf.argmax(output_logits,1), tf.argmax(y,1), name = 'correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)
global_step = 0

num_tr_iter = int(len(y_train)/batch_size)
for epoch in range(epochs):
    print('Training epoch:{}'.format(epoch+1))
    x_train, y_train = randomize(x_train,y_train)
    for iteration in range(num_tr_iter):
        global_step += 1
        start = iteration*batch_size
        end = (iteration+1) * batch_size
        x_batch,y_batch = get_next_batch(x_train,y_train,start,end)
        x_batch = x_batch.reshape((batch_size, timesteps, num_input))
        
        feed_dict_batch = {x: x_batch, y:y_batch}
        sess.run(optimizer, feed_dict=feed_dict_batch)
        
        if iteration % display_freq == 0:
            loss_batch, acc_batch = sess.run([loss, accuracy], feed_dict=feed_dict_batch)
            
            print("iter {0:3d}:\t Loss={1:.2f},\t Training Accuracy={2:.01%}".format(iteration, loss_batch, acc_batch))
            
    feed_dict_valid = {x: x_valid[:1000].reshape((-1, timesteps, num_input)), y:y_valid[:1000]}
    loss_valid, acc_valid = sess.run([loss,accuracy], feed_dict=feed_dict_valid)
    print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:01%}".format(epoch+1,loss_valid,acc_valid))
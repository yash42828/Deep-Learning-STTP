from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

import tensorflow as tf

learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

n_hidden_1 = 512
n_hidden_2 = 352
n_hidden_3 = 256
n_input = 784
n_classes = 10

X = tf.placeholder("float",[None,n_input])
Y = tf.placeholder("float",[None,n_classes])
#print(X.shape)

weights = {
        'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
        'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
        #'h3':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
        'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}
#print(weights['h1'])
biases = {
        'b1':tf.Variable(tf.random_normal([n_hidden_1])),
        'b2':tf.Variable(tf.random_normal([n_hidden_2])),       
        #'b3':tf.Variable(tf.random_normal([n_hidden_3])),
        'out':tf.Variable(tf.random_normal([n_classes]))       
}
#print(biases['b1'])
def multilayer_perceptron(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']), biases['b2'])
    #layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

logits = multilayer_perceptron(X)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                           Y: batch_y})
    
            avg_cost += c/total_batch
        
        if epoch % display_step == 0:
            print("Epoch:",'%04d' %(epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization finished")
    
    pred = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:",accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
    
    
    
    
    
    
    
    
    
    
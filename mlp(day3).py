from PIL import Image
from numpy import *
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import tensorflow as tf

no_of_images = 100
size = 100 * 100
learning_rate = 0.0001
training_epochs = 20
batch_size = 20
display_step = 1
n_hidden_1 = 512 # 1st layer number of neurons
n_hidden_2 = 512 # 2nd layer number of neurons
n_input = size # MNIST data input (img shape: 28*28)
n_classes = 5 # MNIST total classes (0-9 digits)

dataset = ndarray(shape=(no_of_images, size), dtype=float32)
labels = []
folderPath = "/home/utu/Desktop/test/image.orig"
dataFile = folderPath + "/data.csv"
file = open(dataFile, 'r')
line = file.readline()
i = 0
while line:
    #print(line)
    arr = line.split(',')
    cls = int(arr[1])
    #print(arr[0], ' ', cls)
    line = file.readline()

    filePath = folderPath + '/' + arr[0]
    size = 100, 100
    image = Image.open(filePath).convert('L')
    image = image.resize(size)
    image = array(image).flatten()
    #print(image)
    dataset[i,:] = image
    labels.append(cls)
    i = i + 1

print(dataset)
data_labels = asarray(labels)



records = []
for i in range(no_of_images):
    records.append(i)
   
train_size = int(no_of_images * 0.8)
test_size = int(no_of_images * 0.2)

train_part = random.randint(no_of_images, size = train_size)
test_part = delete(records, train_part)

print(train_part)
print(test_part)

label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(max(labels)+1))
labels = label_binarizer.transform(labels)

#x_train, x_test, y_train, y_test = train_test_split(dataset, data_labels, test_size=0.3)
x_train = dataset[train_part, :]
x_test = dataset[test_part, :]

y_train = labels[train_part]
y_test = labels[test_part]

a = tf.random_uniform(
    [batch_size],
    minval=0,
    maxval=train_size,
    dtype=tf.int32
)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        print(epoch)
        avg_cost = 0.
        total_batch = int(80/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            a1 = sess.run(a)
            #print(a1)
            #print(y_train)
            batch_x = x_train[a1, :]
            batch_y = y_train[a1]
            #print(batch_y)
            # Run optimization op (backprop) and cost op (to get loss value)
            #print(batch_x.shape)
            #print(batch_y.shape)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: x_test, Y: y_test}))

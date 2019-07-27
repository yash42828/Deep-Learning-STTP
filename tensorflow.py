import tensorflow as tf
print(tf.__version__)

a = tf.constant([1,2,3,4])
b = tf.constant(10,shape=(2,3))
print(a)

sess = tf.Session()
print(sess.run(a))
print(sess.run(b))
sess.close()
with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    
    
#Variable
x = tf.constant([1,2,3,4])
y = tf.Variable(x+10)
model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)
    print(y.eval())  #same as see.run
    print(sess.run(y))
    y = y+10
    print(sess.run(y))

    
x = tf.Variable(0)
model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)
    for i in range(5):
        x=x+1
    print(sess.run(x))
    


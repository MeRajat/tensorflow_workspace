import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Initialisation of tf network
# Place holder for input images
# Here None specify the batch size and 28 * 28 specify the grayscale image matrix size
X = tf.placeholder(dtype=tf.float32, shape=[None, 784])

# Create a variable conatiner for weights of NN
W = tf.Variable(tf.zeros([784, 10]))
# bias
b = tf.Variable(tf.zeros([10]))
# Initialize all variables
init = tf.initialize_all_variables()


# create model
# here reshape flattens 28*28 image into a vector of size 784 and calculates applies
# Softmax activation function
# Here -1 signifies the only dimension that will preserve the number of elements
# XX = tf.reshape(X, [-1, 784])
Y = tf.nn.softmax(tf.matmul(X, W) + b)

# Placeholder for correct answer ie. labels
Y_ = tf.placeholder(tf.float32, [None, 10])

# loss function i.e cross entropy
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# % correct answers found
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))

# accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#optimizer
# 0.03 is the learning rate
optimizer = tf.train.GradientDescentOptimizer(0.03)

training_step = optimizer.minimize(loss=cross_entropy)


# Now define graph

sess = tf.Session()
sess.run(init)

for i in  range(10000):
    batch_x, batch_y = mnist.train.next_batch(100)
    # feed data into NN
    train_data = {X : batch_x, Y_ : batch_y}
    sess.run(training_step, feed_dict=train_data)

    a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

    # success on test data
    if i %100 == 0:
        # on every 100th iteration
        test_data = {X: mnist.test.images, Y_: mnist.test.labels}
        a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print("accuracy =>", a)
        print("cross entropy =>", c)
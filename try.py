import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib import rnn


map_fn = tf.map_fn

################################################################################
##                           DATASET GENERATION                               ##
##                                                                            ##
##  The problem we are trying to solve is adding two binary numbers. The      ##
##  numbers are reversed, so that the state of RNN can add the numbers        ##
##  perfectly provided it can learn to store carry in the state. Timestep t   ##
##  corresponds to bit len(number) - t.                                       ##
################################################################################

def as_bytes(num, final_size):
    res = []
    for _ in range(final_size):
        res.append(num % 2)
        num //= 2
    return res

def generate_example(num_bits):
    a = random.randint(0, 2**(num_bits - 1) - 1)
    b = random.randint(0, 2**(num_bits - 1) - 1)
    res = a + b
    return (as_bytes(a,  num_bits),
            as_bytes(b,  num_bits),
            as_bytes(res,num_bits))

def generate_batch(num_bits, batch_size):
    """Generates instance of a problem.
    Returns
    -------
    x: np.array
        two numbers to be added represented by bits.
        shape: b, i, n
        where:
            b is bit index from the end
            i is example idx in batch
            n is one of [0,1] depending for first and
                second summand respectively
    y: np.array
        the result of the addition
        shape: b, i, n
        where:
            b is bit index from the end
            i is example idx in batch
            n is always 0
    """
    x = np.empty((num_bits, batch_size, 2))
    y = np.empty((num_bits, batch_size, 1))

    for i in range(batch_size):
        a, b, r = generate_example(num_bits)
        x[:, i, 0] = a
        x[:, i, 1] = b
        y[:, i, 0] = r
    return x, y


################################################################################
##                           GRAPH DEFINITION                                 ##
################################################################################

INPUT_SIZE    = 2       # 2 bits per timestep
RNN_HIDDEN    = 20
OUTPUT_SIZE   = 1       # 1 bit per timestep
TINY          = 1e-6    # to avoid NaNs in logs
LEARNING_RATE = 0.01

USE_LSTM = True

inputs  = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))  # (time, batch, in)
outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE)) # (time, batch, out)


## Here cell can be any function you want, provided it has two attributes:
#     - cell.zero_state(batch_size, dtype)- tensor which is an initial value
#                                           for state in __call__
#     - cell.__call__(input, state) - function that given input and previous
#                                     state returns tuple (output, state) where
#                                     state is the state passed to the next
#                                     timestep and output is the tensor used
#                                     for infering the output at timestep. For
#                                     example for LSTM, output is just hidden,
#                                     but state is memory + hidden
# Example LSTM cell with learnable zero_state can be found here:
#    https://gist.github.com/nivwusquorum/160d5cf7e1e82c21fad3ebf04f039317
if USE_LSTM:
    cell = rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
else:
    cell = rnn.BasicRNNCell(RNN_HIDDEN)

# Create initial state. Here it is just a constant tensor filled with zeros,
# but in principle it could be a learnable parameter. This is a bit tricky
# to do for LSTM's tuple state, but can be achieved by creating two vector
# Variables, which are then tiled along batch dimension and grouped into tuple.
batch_size    = tf.shape(inputs)[1]
print(batch_size)
initial_state = cell.zero_state(batch_size, tf.float32)

# Given inputs (time, batch, input_size) outputs a tuple
#  - outputs: (time, batch, output_size)  [do not mistake with OUTPUT_SIZE]
#  - states:  (time, batch, hidden_size)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)

# project output from rnn output size to OUTPUT_SIZE. Sometimes it is worth adding
# an extra layer here.
final_projection = lambda x: layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)

# apply projection to every timestep.
predicted_outputs = map_fn(final_projection, rnn_outputs)

# compute elementwise cross entropy.
error = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))
error = tf.reduce_mean(error)

# optimize
train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)

# assuming that absolute difference between output and correct answer is 0.5
# or less we can round it to the correct output.
accuracy = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_outputs) < 0.5, tf.float32))


################################################################################
##                           TRAINING LOOP                                    ##
################################################################################

NUM_BITS = 10
ITERATIONS_PER_EPOCH = 100
BATCH_SIZE = 16

valid_x, valid_y = generate_batch(num_bits=NUM_BITS, batch_size=100)

session = tf.Session()
# For some reason it is our job to do this:
session.run(tf.global_variables_initializer())

for epoch in range(5):
    epoch_error = 0
    for _ in range(ITERATIONS_PER_EPOCH):
        # here train_fn is what triggers backprop. error and accuracy on their
        # own do not trigger the backprop.
        x, y = generate_batch(num_bits=NUM_BITS, batch_size=BATCH_SIZE)
        epoch_error += session.run([error, train_fn], {
            inputs: x,
            outputs: y,
        })[0]
    epoch_error /= ITERATIONS_PER_EPOCH
    valid_accuracy = session.run(accuracy, {
        inputs:  valid_x,
        outputs: valid_y,
    })
    print("Epoch %d, train error: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_error, valid_accuracy * 100.0))
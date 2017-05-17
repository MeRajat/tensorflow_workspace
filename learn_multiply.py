import random
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn


def generate_number():
    x = random.randint(1, sys.maxsize) % 10000000
    y = random.randint(1, sys.maxsize) % 10000000
    res = x * y
    return x, y, res


def generate_batch(batch_size):
    """
        This generates random number and their multiplication as a target
    :param batch_size: 
    :return: x : np.array 
                two numbers that to be multiplied
            y : np.array
                target variable (result after multiplication)
    """
    x = np.empty((1, batch_size, 2))
    y = np.empty((1, batch_size, 1))

    for i in range(batch_size):
        a, b, r = generate_number()
        x[:, i, 0] = a
        x[:, i, 1] = b
        y[:, i, 0] = r
    return x, y


map_fn = tf.map_fn
# define graph

INPUT_SIZE = 2  # 2 numbers per time stamp
RNN_HIDDEN = 20
OUTPUT_SIZE = 1  # 1 number per time-stamp
TINY = 1e-6  # to avoid NaNs
LEARNING_RATE = 0.01  # learning rate

# define placeholders
inputs = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))
outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE))

# Define rnn basic cell

cell = rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)

# define initial state
batch_size = tf.shape(inputs)[1]
initial_state = cell.zero_state(batch_size, tf.float32)

# define rnn_outputs and rnn_states
# inputs (time, batch,  input_sizer) outputs a tuple
# - outputs: (time, batch, output_size)
# - states: (time, batch, hidden_size)
rnn_outputs, rnn_states = cell.zero_state(batch_size, tf.float32)

# project output from rnn output size to OUTPUT_SIZE

final_projection = lambda x: layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.sigmoid)

print(" =>", rnn_outputs.shape)
predicted_outputs = map_fn(final_projection, rnn_outputs)

# compute elementwise cross entropy
error = -(outputs * tf.log(predicted_outputs + TINY)) + (1.0 - outputs) \
                                                        * tf.log(1.0 - predicted_outputs + TINY)

error = tf.reduce_mean(error)

train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)
accuracy = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_outputs) < 0.5, tf.float32))

ITERATION_PER_EPOCH = 100
BATCH_SIZE = 16

valid_x, valid_y = generate_batch(BATCH_SIZE)

session = tf.Session()

session.run(tf.global_variables_initializer())

for epoch in range(1000):
    epoch_error = 0
    for _ in range(ITERATION_PER_EPOCH):
        x, y = generate_batch(BATCH_SIZE)
        epoch_error += session.run([error, train_fn], {
            inputs: x,
            outputs: y,
        })[0]
    epoch_error /= ITERATION_PER_EPOCH
    accuracy = session.run(accuracy, {
        inputs: valid_x,
        outputs: valid_y,
    })
    print("Epoch %d, train_error: %.2f, valid_accuracy: %.1f" % (epoch, epoch_error, accuracy * 100.0))

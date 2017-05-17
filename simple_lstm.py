import random

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib import rnn

map_fn = tf.map_fn


def into_bytes(a, final_size):
    """
        This functions converts number into bytes
    :param a:
    :return: array
    """
    res = []
    for _ in range(final_size):
        res.append(a % 2)
        a = a // 2
    return res


def generate_examples(num_bits):
    """
        generates digit in binary and their sum
    :return: a, b, result
    """
    a = random.randint(0, 2 ** (num_bits - 1) - 1)
    b = random.randint(0, 2 ** (num_bits - 1) - 1)
    res = a + b
    return into_bytes(a, num_bits), into_bytes(b, num_bits), into_bytes(res, num_bits)


def generate_batch(batch_size, num_bits):
    """
        this function generate batches of size batch size
    :param batch_size: int
    :param num_bits: int
    :return: X : np.array inputs,
             shape : (b,i, n)
             Y : np.array targets(sum)
    """
    X = np.empty((num_bits, batch_size, 2))
    Y = np.empty((num_bits, batch_size, 1))

    for i in range(num_bits):
        a, b, r = generate_examples(num_bits)
        X[:, i, 0] = a
        X[:, i, 1] = b
        Y[:, i, 0] = r

    return X, Y


"""
    Graph Defination
"""
# define constants
INPUT_SIZE = 2
RNN_HIDDEN = 20
OP_SIZE = 1
TINY = 1e-6
LEARNING_RATE = 0.01

# define placeholders


inputs = tf.placeholder(dtype=tf.float32, shape=(None, None, INPUT_SIZE))
outputs = tf.placeholder(dtype=tf.float32, shape=(None, None, OP_SIZE))

# define basic lstm cell
cell = rnn.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)

batch_size = tf.shape(inputs)[1]
inital_state = cell.zero_state(batch_size, tf.float32)

rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=inital_state, time_major=True)

final_projection = lambda x: layers.linear(x, num_outputs=OP_SIZE, activation_fn=tf.nn.sigmoid)

predicted_outputs = map_fn(final_projection, rnn_outputs)
print("predicted_op  =>", predicted_outputs)

error = -(outputs * tf.log(predicted_outputs + TINY) + (1.0 - outputs) * tf.log(1.0 - predicted_outputs + TINY))

error = tf.reduce_mean(error)

train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)

accuracy = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_outputs) < 0.5, tf.float32))

# training loop

NUM_BITS = 10

ITERATION_PER_LOOP = 100

BATCH_SIZE = 16

valid_x, valid_y = generate_batch(num_bits=NUM_BITS, batch_size=100)

session = tf.Session()

session.run(tf.initialize_all_variables())

for epoch in range(1000):
    epoch_error = 0
    for _ in range(ITERATION_PER_LOOP):
        x, y = generate_batch(num_bits=NUM_BITS, batch_size=BATCH_SIZE)
        epoch_error += session.run([error, train_fn], {
            inputs: x,
            outputs: y
        })[0]

    epoch_error /= ITERATION_PER_LOOP
    x, y = generate_batch(num_bits=NUM_BITS, batch_size=BATCH_SIZE)
    valid_accuracy = session.run(accuracy, {
        inputs: x,
        outputs: y
    })

    print("epoch %d, train_error %.2f, valid_accuracy %.1f " % (epoch, epoch_error, valid_accuracy * 100))

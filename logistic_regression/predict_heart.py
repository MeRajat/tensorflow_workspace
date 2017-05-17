import tensorflow as tf
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# read data

df = pd.read_csv("data/chdage.csv", header=0)
print(df.head)
plt.figure()
plt.scatter(df['Age'], df['Chd'])


# define model
# params for lr

LEARNING_RATE = 0.8
BATCH_SIZE = 100
DISPLAY_STEP = 2

X = tf.placeholder(tf.float32, [None, 1]) # place holder for 1d data
Y = tf.placeholder(tf.float32, [None, 2]) # place holder for classes

W = tf.Variable(tf.zeros([1,2]))
b = tf.Variable(tf.zeros([2]))

activation = tf.nn.softmax(tf.matmul(X, W) + b)

tf.reduce_mean


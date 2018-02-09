import tensorflow as tf
import numpy as np
from random import randint


# Placeholders
x_input = tf.placeholder(shape=[None], dtype=tf.float32, name='x_input')
y_actual = tf.placeholder(shape=[None], dtype=tf.float32, name='y_input')
learning_rate=0.01
# y = x*W+b
W = tf.Variable(-5.0)

y_predict = x_input*W

cost = tf.losses.mean_squared_error(labels=y_actual, predictions=y_predict)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(50):
        for i in range(100):
            num = randint(1,50)
            sess.run(optimizer, feed_dict={x_input:[num], y_actual:[num*5]})
        print(sess.run(W))

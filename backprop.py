import tensorflow as tf
import numpy as np
from random import randint

x = tf.placeholder(shape=[None], dtype=tf.float32)
y_actual = tf.placeholder(shape=[None], dtype=tf.float32)
W = tf.Variable(-5.0)
y_predict = x*W
learning_rate=0.01

cost = tf.losses.mean_squared_error(labels=y_actual, predictions=y_predict)
print("Cost: ", cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        for j in range(100):
            num = randint(1,50)
            sess.run(optimizer, feed_dict={x:[num], y_actual:[num*5]})
        print(sess.run(W))

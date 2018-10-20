import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./', one_hot=True)

# Placeholders
x = tf.placeholder(shape=(None,784), dtype=tf.float32, name='x_input')
y = tf.placeholder(shape=(None,10), dtype=tf.float32, name='target')

# Hyperparameters
learning_rate = 0.0045
num_epochs = 3
batch_size = 64

def get_weights(shape):
    return tf.Variable(tf.random_normal(shape))
def get_biases(shape):
    return tf.Variable(tf.random_normal(shape))
def conv2d(x, W):
    return tf.nn.conv2d(x ,W, strides = [1, 1, 1, 1], padding = "SAME")
def max_pool(x):
    return tf.nn.max_pool(x , ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")


W_conv1 = get_weights([5, 5, 1, 32])
W_conv2 = get_weights([5, 5, 32, 64])
W_fc = get_weights([7*7*64, 1024])
W_out = get_weights([1024, 10])

b_conv1 = get_biases([32])
b_conv2 = get_biases([64])
b_fc = get_biases([1024])
b_out = get_biases([10])


data = tf.reshape(x, shape=[-1, 28, 28, 1])
conv1 = tf.nn.relu(conv2d(data, W_conv1)+b_conv1)
conv1= max_pool(conv1)

conv2 = tf.nn.relu(conv2d(conv1, W_conv2)+b_conv2)
conv2 = max_pool(conv2)

fully_connected = tf.reshape(conv2, [-1, 7*7*64])
fully_connected = tf.nn.relu(tf.matmul(fully_connected, W_fc)+b_fc)
fully_connected = tf.nn.dropout(fully_connected, 0.8)

logits = tf.matmul(fully_connected, W_out) + b_out

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))
#tf.summary.scalar("Cost", cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))
tf.summary.scalar("Accuracy", accuracy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./', sess.graph)

    print("Training ...")

    for epoch in range(num_epochs):
        epoch_loss = 0
        for j in range(100):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
            # if j % 10 == 0:
            #     summary ,_ ,c = sess.run([merged, optimizer, cost], feed_dict = {x : epoch_x, y : epoch_y})
            #     summary_writer.add_summary(summary, (epoch*100)+j)
            # else:
            #     _, c = sess.run([optimizer, cost], feed_dict = {x : epoch_x, y : epoch_y})
            epoch_loss += c
        print("Epoch: ",epoch,", Loss: ",epoch_loss)
    acc = accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels})
    print("Accuracy : ", acc)

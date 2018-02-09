import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('./', one_hot=True)

# Placeholders
x = tf.placeholder(shape=[None,784], dtype=tf.float32, name= 'input_x')
y = tf.placeholder(shape=[None,10], dtype=tf.float32, name='target')

# Hyperparameters
layer1_nodes = 64
layer2_nodes = 128
num_classes = 10
batch_size = 64
num_epochs = 5
learning_rate = 0.001

# Defining Network Architecture
def get_weights(shape):
    return tf.Variable(tf.random_normal(shape))
def get_biases(shape):
    return tf.Variable(tf.random_normal(shape))

W_1 = get_weights([784, layer1_nodes])
W_2 = get_weights([layer1_nodes, layer2_nodes])
W_out = get_weights([layer2_nodes, num_classes])

b_1 = get_biases([layer1_nodes])
b_2 = get_biases([layer2_nodes])
b_out = get_biases([num_classes])

layer1 = tf.matmul(x,W_1)+b_1
layer2 = tf.matmul(layer1, W_2) + b_2
layer_out = tf.matmul(layer2, W_out) + b_out

prediction = layer_out

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

# Launch graph and run optimizer
with tf.Session() as sess:          # Launching the default session
    sess.run(tf.global_variables_initializer())
    epoch_x, epoch_y = mnist.train.next_batch(batch_size)
    print("Training model..")
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(100):
            _, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
            epoch_loss +=c
        print("Epoch: ",epoch," Loss: ", epoch_loss)
    acc = accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels})
    print("Accuracy : ", acc)

# import Libraries
from __future__ import print_function
import numpy as np
import tensorflow as tf
import pickle
from numpy.random import permutation

# =======================================
#           Load Dataset
# =======================================
pickle_file = 'Image_data.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

# ==========================================
#   Define Global Variables
# ==========================================
image_size = 100


def next_batch(dataset, label, batch_size):
    perm = permutation(len(dataset))
    dataset = dataset[perm]
    label = label[perm]
    return dataset[:batch_size], label[:batch_size]

# Set up Tensorflow Interactive session
sess = tf.InteractiveSession()

# Set up input and target variable placeholders
x = tf.placeholder(np.float32, shape=[None, image_size, image_size])
y_ = tf.placeholder(np.float32, shape=[None, 3])


# Weight and Bias Initialization function
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# define convolution and pooling function
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# ================================================================
# Construct the multilayer convolution neural network
# ================================================================
# first layer
W_conv1 = weight_variable([5, 5, 1, 16])
b_conv1 = bias_variable([16])
x_image = tf.reshape(x, [-1, 100, 100, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second layer
W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# classification layer
W_fc1 = weight_variable([25 * 25 * 32, 512])
b_fc1 = bias_variable([512])
h_pool2_flat = tf.reshape(h_pool2, [-1, 25*25*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# define dropout before readout / output layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout / output layer
# first: length output
W_fc2 = weight_variable([512, 3])
b_fc2 = bias_variable([3])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
# y_label = y_[:, 0]
y_label = y_


# Define Training and Evaluation Method
def mean_max_entropy(y_conv, y_):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

batch_size = 32

cross_entropy = mean_max_entropy(y_conv, y_)


# define optimizer
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.05, global_step, 5000, 0.8)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

# define accuracy
acc_bool = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(acc_bool, tf.float32))

# Train & Evaluation
sess.run(tf.initialize_all_variables())
for i in range(600):
    batch = next_batch(train_dataset, train_labels, batch_size)
    if i % 50 == 0:
        # print(batch[0].shape, type(batch[0]))
        # print(batch[1].shape, type(batch[1]))

        loss = cross_entropy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, Loss on Train Set  %g" % (i, loss))

        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g %%" % (i, 100.0 * train_accuracy))

        valid_batch = next_batch(valid_dataset, valid_labels, 1000)
        valid_accuracy = accuracy.eval(feed_dict={
            x: valid_batch[0], y_: valid_batch[1], keep_prob: 1.0})
        print("step %d, Validation accuracy %g %%" % (i, 100.0 * valid_accuracy))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

test_batch = next_batch(test_dataset, test_labels, 1000)
print("test accuracy %g" % accuracy.eval(feed_dict={
    x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))


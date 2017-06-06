"""
CNN_feature_extractor.py

1. Train a CNN for sleep stage classification

2. Extract features at final fully connected layer for LSTM training

created by Tang Yun
June, 2017

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# User defined Classes/methods
from myUtils import prepare_train_valid_data
from myUtils import prepare_test_data
from myUtils import next_batch
from myUtils import label_counter
from myUtils import simple_msg_email
from CNN_configuration import CNNConfiguration

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="2" # 3 is can change to 0-3



tf.logging.set_verbosity(tf.logging.INFO)

# ------------------------------------------------
#   Read Matlab Data Files ".mat"
# ------------------------------------------------
# note that for training data, augmented version is loaded
train_data, train_label, valid_data, valid_label = prepare_train_valid_data()

# ----------------------------------------------
# Construct CNN model
# -----------------------------------------------
# Load Configuration
config = CNNConfiguration()

# Define TF graph input
x = tf.placeholder(tf.float32, [None, config.input_size, config.num_channel])
y = tf.placeholder(tf.float32, [None, config.num_class])
keep_prob = tf.placeholder(tf.float32)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, 3000, 0.9, staircase=True)


# Create Wrappers for reuse
def conv1d(features, w, b, strides=config.conv_stride):

    features = tf.nn.conv1d(value=features, filters=w, stride=strides, padding=config.padding, data_format="NHWC")
    features = tf.nn.bias_add(features, b)
    return config.conv_activation(features)


def max_pool1d(features, k=config.pool_size, s=config.pool_stride):

    return tf.layers.max_pooling1d(inputs=features, pool_size=k, strides=s)


# Create Model
def conv_net(features, weight, bias, dropout=config.drop_out):

    # Conv Layer 1
    conv1 = conv1d(features, weight['wc1'], bias['bc1'])
    # Max Pooling
    conv1 = max_pool1d(conv1)
    # batch normalization
    norm1 = tf.layers.batch_normalization(conv1, axis=1)

    # Conv Layer 2
    conv2 = conv1d(norm1, weight['wc2'], bias['bc2'])
    # Max Pooling
    conv2 = max_pool1d(conv2)
    # batch normalization
    norm2 = tf.layers.batch_normalization(conv2, axis=1)

    # Fully Connected Layer 1
    fc1 = tf.reshape(norm2, [-1, weight['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weight['wd1']), bias['bd1'])
    fc1 = config.dense_activation(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Fully Connected Layer 2
    # fc2 = tf.add(tf.matmul(fc1, weight['wd2']), bias['bd2'])
    # fc2 = config.dense_activation(fc2)

    # Output Layer
    out = tf.add(tf.matmul(fc1, weight['out']), bias['out'])
    return out


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

# Store Layers weights and Biases
weights = {
    # 1d conv kernel = 8, 4 input channel to 64 output channel
    # 'wc1': tf.Variable(tf.random_normal([config.kernel_size, config.num_channel, config.conv_1_channel])),
    'wc1': weight_variable([config.kernel_size, config.num_channel, config.conv_1_channel]),

    # 1d conv kernel = 8, 32 input channel to 128 output channel
    # 'wc2': tf.Variable(tf.random_normal([config.kernel_size, config.conv_1_channel, config.conv_2_channel])),
    'wc2': weight_variable([config.kernel_size, config.conv_1_channel, config.conv_2_channel]),

    # densely connected layer, after pooling twice, feature size becomes 3840 / (8 * 8) = 60,
    # 'wd1': tf.Variable(tf.random_normal([int(config.input_size / (config.pool_size ** 2) * config.conv_2_channel),
    #                                     config.dense_1])),
    'wd1': weight_variable(([int(config.input_size / (config.pool_size ** 2) * config.conv_2_channel),
                             config.dense_1])),

    # densely connected layer 2
    # 'wd2': tf.Variable(tf.random_normal([config.dense_1, config.dense_2])),
    'wd2': weight_variable([config.dense_1, config.dense_2]),

    # output layer
    # 'out': tf.Variable(tf.random_normal([config.dense_1, config.num_class]))
    'out': weight_variable([config.dense_1, config.num_class])
}

biases = {
    'bc1': bias_variable([config.conv_1_channel]),
    'bc2': bias_variable([config.conv_2_channel]),
    'bd1': bias_variable([config.dense_1]),
    'bd2': bias_variable([config.dense_2]),
    'out': bias_variable([config.num_class])
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Saver for saving and restoring models
saver = tf.train.Saver()

# initializing the variables
init = tf.global_variables_initializer()

# launch the graph
with tf.Session() as sess:

    try:
        # saver.restore(sess, '/data2/pengfei_data/CNN_stager/CNN_ckpt/model')
        saver.restore(sess, '/tmp/CNN_model')
    except:
        print("Model checkpoint not found, creating new session...")
        sess.run(init)

    # keep training until reach max iterations
    while sess.run(global_step) < config.max_epochs:
        batch_x, batch_y = next_batch(train_data, train_label, batch_size=config.batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x,
                                       y: batch_y,
                                       keep_prob: config.drop_out})

        # logging
        if sess.run(global_step) % config.display_step == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})

            print("\nIter: " + str(sess.run(global_step)))
            print("Train batch Loss =" + "{:.6f}".format(loss) + ", Train Acc = " + "{:.5f}".format(acc))
            label_counter(batch_y)

            batch_x, batch_y = next_batch(valid_data, valid_label, batch_size=config.batch_size)
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})

            print("Valid batch Loss =" + "{:.6f}".format(loss) + ", Valid Acc = " + "{:.5f}".format(acc))
            label_counter(batch_y)

            # saver.save(sess, '/data2/pengfei_data/CNN_stager/CNN_ckpt/model')
            saver.save(sess, '/tmp/CNN_model')


    print("Training Is Done.")

    # Calculate Accuracy for test samples
    test_data, test_label = prepare_test_data()
    batch_x, batch_y = next_batch(valid_data, valid_label, batch_size=512)
    label_counter(batch_y)
    print("Testing Accuracy: ",
          sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.}))

    simple_msg_email()


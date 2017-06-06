from __future__ import absolute_import, division, print_function

import tensorflow as tf


class CNNConfiguration:

    def __init__(self, config_type=None):
        """ Default configuration for CNN """  # if Relu is used as activation function,
        # we probably don't need batch normalization layer
        self.input_size = 3840
        self.num_channel = 4
        self.num_class = 5
        self.max_epochs = 100000
        self.learning_rate = 0.001
        self.kernel_size = 32  # two second kernel
        self.padding = 'SAME'
        self.conv_stride = 1
        self.conv_1_channel = 128
        self.conv_2_channel = 256
        self.conv_activation = tf.nn.tanh
        self.dense_1 = 1024
        self.drop_out = 0.5
        self.dense_2 = 512
        self.dense_activation = tf.nn.relu
        self.batch_size = 128
        self.pool_size = 8
        self.pool_stride = 8
        self.display_step = 50

        if not config_type:
            pass
        else:
            raise ValueError('Undefined Config_type, please check the value')


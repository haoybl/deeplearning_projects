"""
LSTM_SleepStage_Classifier_main.py

Main Script to construct RNN/LSTM to perform continuous sleep stage classification tasks on PSG data


Created by Tang Yun

"""

# Utility Libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

# Computation Libraries
import numpy as np
import tensorflow as tf

# Self-Defined Libraries
from PSG_data_reader import prepare_train_valid_data

# ----------------------------------------
# Define Flags for command line execution
# ----------------------------------------
flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model", "default", "A type of model. Possible options are: default")
flags.DEFINE_string("data_path", os.path.dirname(os.path.realpath(__file__)), "Where the training/test data is stored")
flags.DEFINE_string("save_path", "/tmp/lstm_SleepStage_classifier/", "Model output directory")
flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class DefaultConfig(object):
    """Default Configuration for LSTM model"""
    init_scale = 0.1        # initial scale for params, random uniform distribution between [-init_scale, init_scale]
    learning_rate = 1.0     # initial learning rate, which decreases after iteration exceeds max_epoch
    max_grad_norm = 5       # control gradient expansion, used to limit gradient L2 norm
    num_layers = 3          # number of layers of each LSTM cell
    num_steps = 5           # number of sleep epochs (30 sec/epoch) included in a single classification task
    hidden_size = 200       # number of neurons in each hidden layer
    max_epoch = 4           # epoch < max_epoch, lr_decay = 1, else lr_decay decreases
    max_max_epoch = 13      # number of iterations of a single recording
    keep_prob = 1.0         # dropout probability to mitigate overfitting
    lr_decay = 0.5          # learning rate decay
    batch_size = 20         # batch size for every training epoch
    class_size = 5          # number of total different sleep stages, WAKE, N1, N2, N3, REM


class StagerModel(object):
    """ Core LSTM model """

    def __init__(self, is_training, config, input_):
        self._input = input_

        batch_size = input_.batch_size
        num_steps =input_.num_steps
        hidden_size = config.hidden_size
        class_size = config.class_size








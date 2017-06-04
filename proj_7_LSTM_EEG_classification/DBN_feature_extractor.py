"""
DBN_feature_extractor.py

1. Train a DBN for sleep stage classification

2. Extract features at final fully connected layer for LSTM training

created by Tang Yun
June, 2017

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from myUtils import prepare_train_valid_data

tf.logging.set_verbosity(tf.logging.INFO)


# ------------------------------------------------
#   Read Matlab Data Files ".mat"
# ------------------------------------------------
train_data, train_label, valid_data, valid_label = prepare_train_valid_data()

# ------------------------------------------------------
# Construct Deep Belief Network Unsupervised feature extraction
# ------------------------------------------------------

# Define RBM class
#class RBM(obj)


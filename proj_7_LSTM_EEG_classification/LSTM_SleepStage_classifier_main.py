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
from myUtils import prepare_train_valid_data






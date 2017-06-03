"""
myRBM.py

Define a RBM class for constructing Deep Belief network

Original code comes from tutorial script ML0120EN-5.2-Review-DBNMNIST,
of online course : Deep Learning with TensorFlow at https://courses.cognitiveclass.ai/

"""


from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

# ------------------------------------------
# Define the basic RBM cell
# ------------------------------------------
class RBM(object):

    def __init__(self, input_size, output_size):
        # Define the hyperparameters
        self._input_size = input_size
        self._output_size = output_size
        self.epochs = 5   # Amount of training iterations
        self.learning_rate = 1.0
        self.batchsize = 100  # number of data samples used per training epoch

        # Initialize Weights and biases as matrices full of zeros
        self.w = np.zeros([input_size, output_size], np.float32)
        self.hb = np.zeros([output_size], np.float32)
        self.vb = np.zeros([input_size], np.float32)

    # Fits the result from the weighted visible layer plus the bias into a sigmoid curve



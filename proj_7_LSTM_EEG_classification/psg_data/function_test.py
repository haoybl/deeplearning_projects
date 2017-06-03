from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import h5py

class test_class(object):

    A = 3

test_class_obj = test_class
print(test_class_obj.A)
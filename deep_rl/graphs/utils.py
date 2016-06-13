from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def get_vars_from_scope(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

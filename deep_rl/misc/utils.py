from __future__ import absolute_import, division, print_function

import tensorflow as tf


def get_vars_from_scope(scope, graph=None):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def first_in_collection(name):
    return tf.get_collection(name)[0]


def kl_divergence(old_probs, new_probs):
    with tf.op_scope([old_probs, new_probs], 'kl_divergence'):
        return tf.reduce_sum(old_probs *
                             (tf.log(new_probs) - tf.log(old_probs), -1))


def likelihood_ratio(x, old_probs, new_probs):
    """
    `x` is a one-hot 2D Tensor
    """
    with tf.op_scope([x, old_probs, new_probs], 'likelihood_ratio'):
        return tf.reduce_sum(x * old_probs, 1) / tf.reduce_sum(x * new_probs,
                                                               1)


def entropy(probs):
    with tf.op_scope([probs], 'entropy'):
        return -tf.reduce_sum(probs * tf.log(probs), 1)

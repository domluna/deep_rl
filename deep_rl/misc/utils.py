from __future__ import absolute_import, division, print_function

import tensorflow as tf
from scipy.signal import lfilter


def slice_2d(x, inds0, inds1):
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(x), tf.int64)
    ncols = shape[1]
    x_flat = tf.reshape(x, [-1])
    return tf.gather(x_flat, inds0 * ncols + inds1)


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


def discount(x, gamma):
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

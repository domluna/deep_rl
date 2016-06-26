from __future__ import absolute_import, division, print_function

import tensorflow as tf

from deep_rl.misc import slice_2d


def create_a3c_graph(input_shape, n_action, model, opt, beta=None, name='a3c'):
    """
    Implements Actor Critic Model (A3C)

    Returns a dictionary of Tensorflow graph operations to be with a
    tf.Session instance.

    Args:
        n_action: A `int`. Number of actions agent can do.
        model: The Tensorflow model
        opt: A `tf.train.Optimizer`.
        beta: A `float`. Regularization term for the entropy of the policy model.
        If beta is `None` no regularization will be added.
    """
    actions = tf.placeholder(tf.int32, shape=(None))
    returns = tf.placeholder(tf.float32, shape=(None))
    states = tf.placeholder(tf.float32, shape=input_shape)

    tf.add_to_collection("actions", actions)
    tf.add_to_collection("returns", returns)
    tf.add_to_collection("states", states)

    probs, value = model(states, n_action)

    tf.add_to_collection("policy", probs)
    tf.add_to_collection("value", value)

    N = tf.shape(states)[0]
    p_vals = slice_2d(probs, tf.range(0, N), actions)
    surr_loss = -tf.log(p_vals + 1e-8)

    policy_loss = surr_loss * (returns - value)
    if beta:
        policy_loss += beta * (-tf.reduce_sum(probs * tf.log(probs + 1e-8), 1))
    value_loss = tf.square(returns - value)
    loss = tf.reduce_mean(policy_loss + value_loss)

    train_op = opt.minimize(loss)

    tf.add_to_collection("loss", loss)
    tf.add_to_collection("train_op", train_op)

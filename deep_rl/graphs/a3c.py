from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers

from deep_rl.misc import slice_2d, get_vars_from_scope


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
    policy_in = tf.placeholder(tf.float32, shape=input_shape)
    value_in = tf.placeholder(tf.float32, shape=input_shape)

    tf.add_to_collection("actions", actions)
    tf.add_to_collection("returns", returns)
    tf.add_to_collection("policy_in", policy_in)
    tf.add_to_collection("value_in", value_in)

    with tf.variable_scope('actor'):
        pnn = model(policy_in)
        probs = tf.nn.softmax(layers.fully_connected(pnn, n_action))
    with tf.variable_scope('critic'):
        v_out = model(value_in)
        value = layers.fully_connected(v_out, 1)

    tf.add_to_collection("policy_out", probs)
    tf.add_to_collection("value_out", value)

    actor_vars = get_vars_from_scope('actor')
    critic_vars = get_vars_from_scope('critic')

    N = tf.shape(states)[0]
    p_vals = slice_2d(probs, tf.range(0, N), actions)
    surr_loss = tf.log(p_vals + 1e-8)

    policy_loss = -surr_loss * (returns - value)
    if beta:
        policy_loss += beta * (-tf.reduce_sum(probs * tf.log(probs + 1e-8), 1))
    policy_loss = tf.reduce_mean(policy_loss, name="policy_loss")
    value_loss = tf.reduce_mean(tf.square(returns - value), name="value_loss")

    policy_train_op = opt.minimize(policy_loss, var_list=actor_vars)
    value_train_op = opt.minimize(value_loss, var_list=critic_vars)

    tf.add_to_collection("policy_loss", policy_loss)
    tf.add_to_collection("value_loss", value_loss)
    tf.add_to_collection("policy_train_op", policy_train_op)
    tf.add_to_collection("value_train_op", value_train_op)

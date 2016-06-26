from __future__ import absolute_import, division, print_function

import tensorflow as tf

from deep_rl.misc.utils import get_vars_from_scope, slice_2d


def create_dqn_graph(n_action, model, opt, gamma=0.99):
    """
    Implements Deep Q-Learning

    if terminal:
        y = r
    else:
        y = r + gamma * max_a' Q(s', a', theta-)

    L = (y - Q(s, a; theta)) ** 2
    """
    actions = tf.placeholder(tf.int32)
    terminals = tf.placeholder(tf.bool)
    rewards = tf.placeholder(tf.float32)

    with tf.variable_scope('policy'):
        p_in, p_out = model()
    with tf.variable_scope('target'):
        t_in, t_out = model()

    p_vars = get_vars_from_scope('policy')
    t_vars = get_vars_from_scope('target')

    mask = (tf.cast(tf.logical_not(terminals), tf.float32))

    y = rewards + mask * gamma * tf.reduce_max(t_out, 1)

    N = tf.shape(p_in)[0]
    policy_probs = tf.nn.softmax(p_out)
    p_vals = slice_2d(policy_probs, tf.range(0, N), actions)

    loss_op = tf.reduce_mean(tf.square(y - p_vals))
    train_op = opt.minimize(loss_op, var_list=p_vars)
    update_targets_op = [tf.assign(tv, pv) for (tv, pv) in zip(t_vars, p_vars)]

    return dict(
        # inputs
        policy_input=p_in,
        target_input=t_in,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        # outputs
        policy_qvals=p_out,
        target_qvals=t_out,
        policy_probs=policy_probs,
        loss_op=loss_op,
        train_op=train_op,
        # misc
        update_targets_op=update_targets_op)

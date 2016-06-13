from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from .utils import get_vars_from_scope

def create_dqn_graph(n_action, model, opt, gamma=0.99):
    """
    Implements DQN

    if terminal:
        y = r
    else:
        y = r + gamma * max_a' Q(s', a', theta-)
    L = (y - Q(s, a; theta)) ** 2
    """
    actions = tf.placeholder(tf.int32)
    terminals = tf.placeholder(tf.bool)
    rewards = tf.placeholder(tf.float32)

    p_in, p_out = model(name='policy-model')
    t_in, t_out = model(name='target-model')
    p_vars = get_vars_from_scope('policy-model')
    t_vars = get_vars_from_scope('target-model')

    terminal_floats = (tf.cast(tf.logical_not(terminals), tf.float32))

    a = tf.one_hot(actions, depth=n_action, on_value=1.0, off_value=0.0)
    y = rewards + terminal_floats * gamma * tf.reduce_max(t_out, 1)

    policy_probs = tf.nn.softmax(p_out)
    loss_op = tf.reduce_mean(tf.square(y - tf.reduce_sum(p_out * a, 1)))
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
            update_targets_op=update_targets_op
            )

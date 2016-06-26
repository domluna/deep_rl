from __future__ import absolute_import, division, print_function

import tensorflow as tf

from deep_rl.misc.utils import get_vars_from_scope, slice_2d


def create_vpg_graph(n_action, policy_model, value_model, policy_opt, value_opt):
    """
    Implements Vanilla Policy Gradient
    """
    actions = tf.placeholder(tf.int32, shape=(None), name="actions")
    advantages = tf.placeholder(tf.float32, shape=(None), name="advantages")
    returns = tf.placeholder(tf.float32, shape=(None), name="returns")

    with tf.variable_scope('policy'):
        p_input, probs = policy_model()
    with tf.variable_scope('value'):
        v_input, value = value_model()

    p_vars = get_vars_from_scope('policy')
    v_vars = get_vars_from_scope('value')

    N = tf.shape(p_input)[0]
    p_vals = slice_2d(probs, tf.range(0, N), actions)
    surr_loss = -tf.log(p_vals)

    pf_loss_op = tf.reduce_mean(surr_loss * advantages, name="pf_loss_op")
    pf_train_op = policy_opt.minimize(pf_loss_op, var_list=p_vars, name="pf_train_op")

    vf_loss_op = tf.reduce_mean((value - returns)**2)
    vf_train_op = value_opt.minimize(vf_loss_op, var_list=v_vars, name="vf_train_op")

    return dict(actions=actions,
                advantages=advantages,
                returns=returns,
                policy_input=p_input,
                probs=probs,
                value_input=v_input,
                value=value,
                policy_train_op=pf_train_op,
                value_train_op=vf_train_op)

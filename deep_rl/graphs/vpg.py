from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from .utils import get_vars_from_scope

def create_vpg_graph(n_action, policy_model, value_model, policy_opt, value_opt):
    """
    Implements Vanilla Policy Gradient
    """
    actions = tf.placeholder(tf.int32, shape=(None), name="actions")
    advantages = tf.placeholder(tf.float32, shape=(None), name="advantages")
    returns = tf.placeholder(tf.float32, shape=(None), name="returns")

    p_input, p_output = policy_model('policy-model')
    v_input, v_output = value_model('value-model')
    p_vars = get_vars_from_scope('policy-model')
    v_vars = get_vars_from_scope('value-model')

    probs = tf.nn.softmax(p_output)
    a = tf.one_hot(actions, depth=n_action, on_value=1.0, off_value=0.0)
    log_lik = tf.log(tf.reduce_sum(probs * a, 1))
    pf_loss_op = -tf.reduce_mean(log_lik * advantages, name="pf_loss_op")
    pf_train_op = policy_opt.minimize(pf_loss_op, var_list=p_vars, name="pf_train_op")

    vf_loss_op = tf.reduce_mean((v_output - returns)**2)
    vf_train_op = value_opt.minimize(vf_loss_op, var_list=v_vars, name="vf_train_op")

    return dict(
        actions=actions,
        advantages=advantages,
        returns=returns,

        policy_input=p_input,
        policy_probs=probs,
        policy_train_op=pf_train_op,

        value_input=v_input,
        value_predict=v_output,
        value_train_op=vf_train_op
    )

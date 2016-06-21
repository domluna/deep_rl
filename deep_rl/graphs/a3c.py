from __future__ import absolute_import, division, print_function

import tensorflow as tf

from deep_rl.misc import get_vars_from_scope


def create_a3c_graph(n_action, model, opt, beta=0.0, name='a3c'):
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
    actions = tf.placeholder(tf.int32, shape=(None), name="actions")
    returns = tf.placeholder(tf.float32, shape=(None), name="returns")

    states, p_out, v_out = model(name=name)

    probs = tf.nn.softmax(p_out)
    a = tf.one_hot(actions, depth=n_action, on_value=1.0, off_value=0.0)
    log_probs = -tf.log(tf.reduce_sum(probs * a, 1))

    policy_loss = log_probs * (returns - v_out)
    if beta != 0.0:
        policy_loss += beta * (-tf.reduce_sum(probs * tf.log(probs), 1))
    value_loss = tf.square(returns - v_out)
    total_loss = tf.reduce_mean(policy_loss + value_loss)

    train_op = opt.minimize(total_loss)
    return dict(states=states,
                actions=actions,
                returns=returns,
                policy_out=probs,
                value_out=v_out,
                train_op=train_op)

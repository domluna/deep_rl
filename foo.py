from __future__ import absolute_import, division, print_function

import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, learn

import gym
from deep_rl.agents import A3CAgent
from deep_rl.graphs import create_a3c_graph
from deep_rl.trajectories import discount, sample_traj
from deep_rl.misc import first_in_collection
from six.moves import range

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("name", "CartPole-v0",
                    "Name of the environment to train/play")
flags.DEFINE_float("gamma", 0.99, "Discount rate")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("epsilon", 0.05, "Exploration rate")
flags.DEFINE_float('beta', 0, "Beta regularization term for A3C")
flags.DEFINE_integer("total_steps", 500000, "Number of iterations")
flags.DEFINE_integer("max_traj_len", 10,
                     "Maximum steps taken during a trajectory")
flags.DEFINE_integer("save_interval", 600, "Interval to save model graph")
flags.DEFINE_integer("eval_interval", 10000, "Interval to evaluate model")
flags.DEFINE_integer("num_threads", 1, "Number of threads or environments to explore concurrently")
flags.DEFINE_bool("render", False, "Render environment during training")
flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_string(
    "outdir", "", "Prefix for monitoring, summary and checkpoint directories")


def sample_policy_action(probs):
    """
    Sample an action from an action probability distribution output by
    the policy network.
    """
    # Subtract a tiny value from probabilities in order to avoid
    # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
    probs = probs - np.finfo(np.float32).epsneg

    histogram = np.random.multinomial(1, probs)
    action_index = int(np.nonzero(histogram)[0])
    return action_index


def simple_nn(states, n_action, hidden_sizes, name='simple_nn'):
    with tf.variable_scope(name):
        hiddens = learn.ops.dnn(states, hidden_sizes, activation=tf.nn.relu)
        probs = tf.nn.softmax(layers.fully_connected(hiddens, n_action))
        value = layers.fully_connected(hiddens, 1)
        return probs, value


np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)
random.seed(FLAGS.seed)

# if FLAGS.outdir:
#     monitor_dir = outdir
#     summary_dir = None
#     checkpoint_dir = None

# env.monitor.start('/tmp/' + FLAGS.name, force=True)

env = gym.make(FLAGS.name)
n_action = env.action_space.n
input_shape = (None, ) + env.observation_space.shape
env.close()


def main(_):
    g = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with g.as_default(), tf.Session(
            config=config) as sess, tf.device('/cpu:0'):
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        model = lambda states, n_action: simple_nn(states, n_action, [20])
        create_a3c_graph(input_shape,
                         n_action,
                         model,
                         opt,
                         beta=None,
                         name='a3c')

        # Finalize the graph! If we're adding ops to it after
        # an error will be thrown.
        sess.run(tf.initialize_all_variables())
        g.finalize()

        global_step = first_in_collection("global_step")
        T = tf.train.global_step(sess, global_step)

        agent = A3CAgent(FLAGS  .name, g, sess, FLAGS.gamma, FLAGS.epsilon,
                         FLAGS.max_traj_len, sample_policy_action, '/tmp/foo', num_threads=FLAGS.num_threads)

        agent.train()

        T = tf.train.global_step(sess, global_step)
        print(T)


        # if t % FLAGS.eval_interval == 0:
        #     total_reward = agent.test()
        #     print("-------------------------")
        #     print("Current step: {}".format(0))
        #     print("Total Reward = {:.2f}".format(total_reward))
        #     print("-------------------------")


if __name__ == "__main__":
    tf.app.run()

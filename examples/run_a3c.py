from __future__ import absolute_import, division, print_function

import random
import threading
import tempfile
import logging

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, learn

import gym
from deep_rl.agents import A3CAgent
from deep_rl.graphs import create_a3c_graph
from deep_rl.trajectories import discount, sample_traj
from deep_rl.envs import EnvWrapper
from deep_rl.misc import first_in_collection, categorical_sample
from six.moves import range

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("name", "CartPole-v0", "Name of the environment to train/play")
flags.DEFINE_float("gamma", 0.99, "Discount rate")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("epsilon", 0.05, "Exploration rate")
flags.DEFINE_float('beta', 0.01, "Beta regularization term for A3C")
flags.DEFINE_integer("max_traj_len", 4, "Maximum steps taken during a trajectory")
flags.DEFINE_integer("save_model_interval", 120, "Interval to save model (seconds)")
flags.DEFINE_integer("save_summaries_interval", 120, "Interval to save summaries (seconds)")
flags.DEFINE_integer("num_threads", 1, "Number of threads or environments to explore concurrently")
flags.DEFINE_integer("exploration_steps", 1, "")
flags.DEFINE_integer("static_steps", 1, "")
flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_string("outdir", "", "Prefix for monitoring, summary and checkpoint directories")
flags.DEFINE_bool("render", False, "Render environment during training")


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


def simple_nn(states, n_action, hidden_sizes):
    hiddens = learn.ops.dnn(states, hidden_sizes, activation=tf.nn.relu)
    probs = tf.nn.softmax(layers.fully_connected(hiddens, n_action))
    value = layers.fully_connected(hiddens, 1)
    return probs, value


outdir = FLAGS.outdir
if outdir == "":
    outdir = tempfile.mkdtemp()

monitor_env = EnvWrapper(FLAGS.name)
monitor_dir = outdir + '/monitor'

n_action = monitor_env.action_space.n
input_shape = (None,) + monitor_env.observation_space.shape

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)
random.seed(FLAGS.seed)
monitor_env.seed(FLAGS.seed)

logging.getLogger().setLevel(logging.DEBUG)

print("Input shape {}".format(input_shape))
print("Number of Actions {}".format(n_action))


def main(_):
    g = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with g.as_default():
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        model = lambda states, n_action: simple_nn(states, n_action, [32])
        create_a3c_graph(input_shape, n_action, model, opt, beta=FLAGS.beta)

        T = tf.Variable(0, trainable=False)
        tf.add_to_collection("global_step", T)

        agent = A3CAgent(g, FLAGS.gamma, FLAGS.epsilon, FLAGS.max_traj_len, categorical_sample)

        sv = tf.train.Supervisor(g,
                                 logdir=outdir,
                                 global_step=T,
                                 save_model_secs=FLAGS.save_model_interval,
                                 save_summaries_secs=FLAGS.save_summaries_interval)

        with sv.managed_session(config=config) as sess:
            try:
                coord = sv.coord
                envs = [EnvWrapper(FLAGS.name) for _ in range(FLAGS.num_threads - 1)]
                envs.insert(0, monitor_env)

                for e in envs:
                    e.monitor.start(monitor_dir, resume=True, video_callable=False)

                threads = [threading.Thread(target=agent.run,
                                            args=(i, sess, coord, envs[i]))
                           for i in range(FLAGS.num_threads)]
                for t in threads:
                    t.start()
                coord.join(threads)
            except Exception as e:
                print("Error training model ...")
                print(e)


if __name__ == "__main__":
    tf.app.run()

"""
DQN
"""
# TODO: make this an example for image atari/doom envs
from __future__ import absolute_import, division, print_function

import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, learn

import gym
from deep_rl.envs import EnvWrapper
from deep_rl.graphs import create_dqn_graph
from deep_rl.memory import Buffer, UniformExperienceReplay

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("name", "CartPole-v0",
                    "Name of the environment to train/play")
flags.DEFINE_float("gamma", 0.99, "Discount rate")
flags.DEFINE_float("epsilon_start", 1.0, "Epsilon start value")
flags.DEFINE_float("epsilon_end", 0.1, "Epsilon end value (after decay)")
flags.DEFINE_float("learning_rate", 0.00025, "Learning rate for Optimizer")
flags.DEFINE_integer("decay_steps", 200000,
                     "Number of steps to train with epsilon decay")
flags.DEFINE_integer("static_steps", 100000,
                     "Number of steps to train with epsilon = epsilon_end")
flags.DEFINE_integer("batch_size", 32,
                     "Number of experiences to sample for training")
flags.DEFINE_integer("replay_capacity", 20000, "Capacity of Experience Replay")
flags.DEFINE_integer(
    "random_start", 1000,
    "Number of experiences to populate Replay with prior to training")
flags.DEFINE_integer("history_len", 4,
                     "Number of observations to be considered a state")
flags.DEFINE_integer("update_targets_interval", 10000,
                     "Number of observations to be considered a state")
flags.DEFINE_bool("render", False, "Render environment during training")
flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_string(
    "outdir", "", "Prefix for monitoring, summary and checkpoint directories")


def simple_model(input_shape, hidden_sizes, n_action, name=None):
    with tf.variable_scope(name or 'simple_model'):
        states = tf.placeholder("float", shape=input_shape)
        hiddens = learn.ops.dnn(states, hidden_sizes, activation=tf.nn.elu)
        return states, layers.fully_connected(hiddens, n_action)


env = EnvWrapper(FLAGS.name)
n_action = env.action_space.n
ob_shape = env.observation_space.shape
ob_dtype = env.observation_space.sample().dtype
input_shape = (None, ) + (np.prod((FLAGS.history_len, ) + ob_shape), )
n_action = env.action_space.n
total_steps = FLAGS.epsilon_decay_steps + FLAGS.epsilon_static_steps


def main(_):
    g = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with g.as_default(), tf.Session(
            config=config) as sess, tf.device('/cpu:0'):
        opt = tf.train.RMSP(learning_rate=FLAGS.learning_rate)
        model = lambda name: simple_model(input_shape, [20], n_action, name=name)
        graph_ops = create_dqn_graph(n_action, model, opt)
        replay = UniformExperienceReplay(FLAGS.replay_capacity,
                                         FLAGS.history_len,
                                         ob_shape,
                                         ob_dtype,
                                         flatten=True)
        buf = Buffer(FLAGS.history_len, flatten=True)
        agent = DQNAgent(env, replay, buf)

        sess.run(tf.initialize_all_variables())
        g.finalize()

        random_start(FLAGS.random_start)

        epsilon_decay_stride = (
            FLAGS.epsilon_start - FLAGS.epsilon_end) / FLAGS.decay_steps
        epsilon = FLAGS.epsilon_start

        sess.run(graph_ops["update_targets_op"])
        t0 = time.time()
        for t in range(1, total_steps + 1):
            action = agent.get_action(graph_ops, sess, epsilon)
            agent.act(action)
            agent.train_batch(graph_ops, sess)

            if t % FLAGS.update_targets_interval == 0:
                sess.run(graph_ops["update_targets_op"])
                ep_reward, action_stats = agent.eval(graph_ops, sess)

                print("-------------------")
                print("Timestep {}".format(t))
                print("Epsilon = {}".format(epsilon))
                print("Episode Reward = {}".format(ep_reward))
                print("Action Stats = {}".format(action_stats))
                print("Time Taken = {:.2f}".format(time.time() - t0))
                print("-------------------")
                t0 = time.time()

            epsilon = max(FLAGS.epsilon_end, epsilon - epsilon_decay_stride)


if __name__ == '__main__':
    tf.app.run()

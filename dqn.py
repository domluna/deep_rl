"""
DQN
"""
# TODO: make this an example for image atari/doom envs
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import gym
import time

from tensorflow.contrib import learn, layers
from deep_rl.memory import UniformExperienceReplay, Buffer
from deep_rl.graphs import create_dqn_graph
from envs import EnvWrapper

def simple_model(input_shape, hidden_sizes, n_action, name=None):
    with tf.variable_scope(name or 'simple_model'):
        states = tf.placeholder("float", shape=input_shape)
        hiddens = learn.ops.dnn(states, hidden_sizes, activation=tf.nn.elu)
        return states, layers.fully_connected(hiddens, n_action)

replay_capacity = 10000
batch_size = 32
history_len = 1

env = EnvWrapper('CartPole-v0')
n_action = env.action_space.n
ob_shape = env.observation_space.shape
input_shape = (None,) + (np.prod((history_len,) + ob_shape),)
n_action = env.action_space.n

g = tf.Graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with g.as_default(), tf.Session(config=config) as sess, tf.device('/cpu:0'):
    opt = tf.train.AdamOptimizer(learning_rate=1e-4)
    model = lambda name: simple_model(input_shape, [40, 40], n_action, name=name)
    graph_ops = dqn_graph(n_action, model, opt)
    replay = UniformExperienceReplay(replay_capacity, batch_size, history_len, ob_shape, np.float32, flatten=True)
    buf = Buffer(history_len, flatten=True)
    agent = DQNAgent(env, replay, buf)

    sess.run(tf.initialize_all_variables())

    g.finalize()

    agent.random_start(500)

    epsilon_start = 1.0
    epsilon_end = 0.1
    decay_steps = 200000
    total_steps = 300000
    epsilon_decay_stride = (epsilon_start - epsilon_end) / decay_steps

    epsilon = epsilon_start

    sess.run(graph_ops["update_targets_op"])
    t0 = time.time()
    for t in range(1, total_steps+1):
        action = agent.get_action(graph_ops, sess, epsilon)
        agent.act(action)
        agent.train_batch(graph_ops, sess)

        if t % 10000 == 0:
            sess.run(graph_ops["update_targets_op"])
            ep_reward, action_stats = agent.eval(graph_ops, sess)

            print("Timestep {}".format(t))
            print("Epsilon {}".format(epsilon))
            print("Episode Reward: {}".format(ep_reward))
            print("Action Stats: {}".format(action_stats))
            print("Time Taken: {:.2f}".format(time.time()-t0))
            print("-------------------")
            t0 = time.time()


        epsilon = max(0.1, epsilon - epsilon_decay_stride)

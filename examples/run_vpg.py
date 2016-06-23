from __future__ import absolute_import, division, print_function

import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, learn

import gym
from deep_rl.graphs import create_vpg_graph
from deep_rl.trajectories import compute_vpg_advantage, sample_traj
from six.moves import range

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("name", "CartPole-v0",
                    "Name of the environment to train/play")
flags.DEFINE_float("gamma", 0.99, "Discount rate")
flags.DEFINE_float("gae_lambda", 1, "Lambda for GAE")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_integer("iters", 100, "Number of iterations")
flags.DEFINE_integer("batch_size", 5000, "Number of timesteps/iteration")
flags.DEFINE_bool("render", False, "Render environment during training")
flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_string(
    "outdir", "", "Prefix for monitoring, summary and checkpoint directories")


def policy_model(input_shape, hidden_sizes, n_action, name='policy_model'):
    states = tf.placeholder("float", shape=input_shape)
    with tf.variable_scope(name):
        hiddens = learn.ops.dnn(states, hidden_sizes, activation=tf.nn.relu)
        return states, layers.fully_connected(hiddens, n_action)


def value_model(input_shape, hidden_sizes, name='value_model'):
    states = tf.placeholder("float", shape=input_shape)
    with tf.variable_scope(name):
        hiddens = learn.ops.dnn(states, hidden_sizes, activation=tf.nn.relu)
        return states, layers.fully_connected(hiddens, 1)


env = gym.make(FLAGS.name)

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)
env.seed(FLAGS.seed)

# if FLAGS.outdir:
#     monitor_dir = outdir
#     summary_dir = None
#     checkpoint_dir = None

# env.monitor.start('/tmp/' + FLAGS.name, force=True)

n_action = env.action_space.n
input_shape = (None, ) + env.observation_space.shape


def main(unused_args):
    g = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with g.as_default(), tf.Session(
            config=config) as sess, tf.device('/cpu:0'):
        pf_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        vf_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        pf_model = lambda name: policy_model(input_shape, [20], n_action, name=name)
        vf_model = lambda name: value_model(input_shape, [20], name=name)
        graph_ops = create_vpg_graph(n_action, pf_model, vf_model, pf_opt,
                                     vf_opt)

        sess.run(tf.initialize_all_variables())

        # Finalize the graph! If we're adding ops to it after
        # and error will be thrown.
        g.finalize()

        actions = graph_ops["actions"]
        advantages = graph_ops["advantages"]
        returns = graph_ops["returns"]

        pf_input = graph_ops["policy_input"]
        pf_train_op = graph_ops["policy_train_op"]
        pf_probs_op = graph_ops["policy_probs"]

        vf_input = graph_ops["value_input"]
        vf_predict_op = graph_ops["value_predict"]
        vf_train_op = graph_ops["value_train_op"]

        pf_action = lambda x: sess.run(pf_probs_op, feed_dict={pf_input: x.reshape(1, -1)})
        vf_predict = lambda x: sess.run(vf_predict_op, feed_dict={vf_input: x})[:, 0]

        for i in range(FLAGS.iters):
            trajs = []
            t0 = time.time()
            timesteps_count = 0
            trajs_count = 0
            while timesteps_count < FLAGS.batch_size:
                t = sample_traj(env,
                                pf_action,
                                max_traj_len=env.spec.timestep_limit)
                trajs.append(t)
                timesteps_count += len(t["actions"])
                trajs_count += 1
            compute_vpg_advantage(trajs, vf_predict, FLAGS.gamma,
                                  FLAGS.gae_lambda)

            all_states = np.concatenate([t["states"] for t in trajs])
            all_acts = np.concatenate([t["actions"] for t in trajs])
            all_rets = np.concatenate([t["returns"] for t in trajs])
            all_advs = np.concatenate([t["advantages"] for t in trajs])

            # train models
            sess.run(vf_train_op,
                     feed_dict={vf_input: all_states,
                                returns: all_rets})
            sess.run(pf_train_op,
                     feed_dict={pf_input: all_states,
                                advantages: all_advs,
                                actions: all_acts})

            reward_sums = np.array([t["rewards_sum"] for t in trajs])
            reward_mean = reward_sums.mean()
            reward_std = reward_sums.std() / np.sqrt(len(reward_sums))

            print("------------------------")
            print("Iteration {}".format(i))
            print("Number of timesteps {}".format(timesteps_count))
            print("Trajectories sampled {}".format(trajs_count))
            print("Max Reward = {:.2f}".format(np.max(reward_sums)))
            print("Average Reward = {:.2f} +- {:.2f}".format(reward_mean,
                                                             reward_std))
            print("Time taken = {:.2f}".format(time.time() - t0))
            print("------------------------")

            # render after each iteration
            if FLAGS.render:
                sample_traj(env,
                            pf_action,
                            max_traj_len=env.spec.timestep_limit,
                            render=True)


if __name__ == "__main__":
    tf.app.run()

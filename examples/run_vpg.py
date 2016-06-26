from __future__ import absolute_import, division, print_function

import time

import numpy as np
import tensorflow as tf
import logging
from tensorflow.contrib import layers, learn

import gym
from deep_rl.graphs import create_vpg_graph
from deep_rl.trajectories import compute_vpg_advantage, sample_traj
from deep_rl.misc import categorical_sample
from six.moves import range

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("name", "CartPole-v0", "Name of the environment to train/play")
flags.DEFINE_float("gamma", 0.99, "Discount rate")
flags.DEFINE_float("gae_lambda", 1, "Lambda for GAE")
flags.DEFINE_float("learning_rate", 0.05, "Learning rate")
flags.DEFINE_integer("batch_size", 2000, "Number of timesteps per batch")
flags.DEFINE_bool("render", False, "Render environment during training")
flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_string("outdir", "", "Prefix for monitoring, summary and checkpoint directories")
flags.DEFINE_integer("save_model_interval", 120, "Interval to save model (seconds)")
flags.DEFINE_integer("save_summaries_interval", 120, "Interval to save summaries (seconds)")


def policy_model(input_shape, hidden_sizes, n_action):
    states = tf.placeholder("float", shape=input_shape)
    hiddens = learn.ops.dnn(states, hidden_sizes, activation=tf.nn.relu)
    return states, tf.nn.softmax(layers.fully_connected(hiddens, n_action))


def value_model(input_shape, hidden_sizes):
    states = tf.placeholder("float", shape=input_shape)
    hiddens = learn.ops.dnn(states, hidden_sizes, activation=tf.nn.relu)
    return states, layers.fully_connected(hiddens, 1)


env = gym.make(FLAGS.name)

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)
env.seed(FLAGS.seed)

n_action = env.action_space.n
input_shape = (None,) + env.observation_space.shape
monitor_dir = FLAGS.outdir + '/monitor'

logging.getLogger().setLevel(logging.DEBUG)

env.monitor.start(monitor_dir, resume=True, video_callable=False)


def main(unused_args):
    g = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with g.as_default(), tf.device('/cpu:0'):
        pf_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        vf_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        pf_model = lambda: policy_model(input_shape, [32], n_action)
        vf_model = lambda: value_model(input_shape, [32])
        graph_ops = create_vpg_graph(n_action, pf_model, vf_model, pf_opt, vf_opt)

        sv = tf.train.Supervisor(g,
                                 logdir=FLAGS.outdir,
                                 save_model_secs=FLAGS.save_model_interval,
                                 save_summaries_secs=FLAGS.save_summaries_interval)

        with sv.managed_session(config=config) as sess:

            actions = graph_ops["actions"]
            advantages = graph_ops["advantages"]
            returns = graph_ops["returns"]

            pf_input = graph_ops["policy_input"]
            pf_train_op = graph_ops["policy_train_op"]
            pf_probs_op = graph_ops["probs"]

            vf_input = graph_ops["value_input"]
            vf_predict_op = graph_ops["value"]
            vf_train_op = graph_ops["value_train_op"]

            def pf_action(x):
                probs = sess.run(pf_probs_op, feed_dict={pf_input: x.reshape(1, -1)})
                return categorical_sample(probs)[0]

            vf_predict = lambda x: sess.run(vf_predict_op, feed_dict={vf_input: x})[:, 0]

            total_steps = 0
            while True:
                trajs = []
                t0 = time.time()
                timesteps_count = 0
                trajs_count = 0
                while timesteps_count < FLAGS.batch_size:
                    t = sample_traj(env, pf_action, max_traj_len=env.spec.timestep_limit)
                    trajs.append(t)
                    timesteps_count += len(t["actions"])
                    trajs_count += 1
                compute_vpg_advantage(trajs, vf_predict, FLAGS.gamma, FLAGS.gae_lambda)

                all_states = np.concatenate([t["states"] for t in trajs])
                all_acts = np.concatenate([t["actions"] for t in trajs])
                all_rets = np.concatenate([t["returns"] for t in trajs])
                all_advs = np.concatenate([t["advantages"] for t in trajs])

                # train models
                sess.run(vf_train_op, feed_dict={vf_input: all_states, returns: all_rets})
                sess.run(pf_train_op,
                         feed_dict={pf_input: all_states,
                                    advantages: all_advs,
                                    actions: all_acts})

                reward_sums = np.array([t["rewards_sum"] for t in trajs])
                reward_mean = reward_sums.mean()
                reward_std = reward_sums.std() / np.sqrt(len(reward_sums))

                total_steps += timesteps_count
                print("------------------------")
                print("Total timesteps {}".format(total_steps))
                print("Number of timesteps {}".format(timesteps_count))
                print("Trajectories sampled {}".format(trajs_count))
                print("Max Reward = {:.2f}".format(np.max(reward_sums)))
                print("Average Reward = {:.2f} +- {:.2f}".format(reward_mean, reward_std))
                print("Time taken = {:.2f}".format(time.time() - t0))
                print("------------------------")

                # render after each iteration
                if FLAGS.render:
                    sample_traj(env, pf_action, max_traj_len=env.spec.timestep_limit, render=True)


if __name__ == "__main__":
    tf.app.run()

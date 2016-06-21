from __future__ import absolute_import, division, print_function

import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, learn

import gym
from deep_rl.graphs import create_a3c_graph
from deep_rl.trajectories import discount, sample_traj
from six.moves import range

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("name", "CartPole-v0",
                    "Name of the environment to train/play")
flags.DEFINE_float("gamma", 0.99, "Discount rate")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
flags.DEFINE_float("epsilon", 0.05, "Exploration rate")
flags.DEFINE_float('beta', 0.02, "Beta regularization term for A3C")
flags.DEFINE_integer("total_steps", 500000, "Number of iterations")
flags.DEFINE_integer("max_traj_len", 5,
                     "Maximum steps taken during a trajectory")
flags.DEFINE_integer("save_interval", 600, "Interval to save model graph")
flags.DEFINE_integer("eval_interval", 10000, "Interval to evaluate model")
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


def simple_nn(input_shape, hidden_sizes, n_action, name='simple_nn'):
    states = tf.placeholder("float", shape=input_shape)
    with tf.variable_scope(name):
        hiddens = learn.ops.dnn(states, hidden_sizes, activation=tf.nn.relu)
        policy = layers.fully_connected(hiddens, n_action)
        value = layers.fully_connected(hiddens, 1)
        return states, policy, value


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
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        create_model = lambda name: simple_nn(input_shape, [20], n_action, name=name)
        graph_ops = create_a3c_graph(n_action,
                                     create_model,
                                     opt,
                                     beta=FLAGS.beta,
                                     name='a3c')

        # Finalize the graph! If we're adding ops to it after
        # and error will be thrown.
        sess.run(tf.initialize_all_variables())
        g.finalize()

        _actions = graph_ops["actions"]
        _returns = graph_ops["returns"]
        _states = graph_ops["states"]

        policy_out = graph_ops["policy_out"]
        value_out = graph_ops["value_out"]
        train_op = graph_ops["train_op"]

        def compute_action(state):
            probs = sess.run(policy_out, feed_dict={_states: state.reshape(1, -1)})[0]
            return sample_policy_action(probs)

        t_start = 0
        actions = []
        states = []
        rewards = []

        state = env.reset()
        for t in range(1, FLAGS.total_steps + 1):
            if random.random() < FLAGS.epsilon:
                action = env.action_space.sample()
            else:
                action = compute_action(state)

            next_state, reward, done, info = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # update params
            if done or t - t_start == FLAGS.max_traj_len:
                last_state = states[-1]
                val = 0
                if not done:
                    val = sess.run(
                        value_out,
                        feed_dict={_states: last_state.reshape(1, -1)})
                rewards.append(val)
                returns = discount(rewards, FLAGS.gamma)[:-1]
                sess.run(train_op,
                         feed_dict={_states: states,
                                    _returns: returns,
                                    _actions: actions})

                actions = []
                states = []
                rewards = []
                t_start = t

            if done:
                state = env.reset()
            else:
                state = next_state

            if t % FLAGS.eval_interval == 0:
                traj = sample_traj(env,
                                   compute_action,
                                   max_traj_len=env.spec.timestep_limit,
                                   render=FLAGS.render)

                total_reward = traj["rewards"].sum()
                print("-------------------------")
                print("Current step: {}".format(t))
                print("Total Reward = {:.2f}".format(total_reward))
                print("-------------------------")

                state = env.reset()


if __name__ == "__main__":
    tf.app.run()

from __future__ import absolute_import, division, print_function

import random

import tensorflow as tf

import gym
from deep_rl.graphs import create_a3c_graph
from deep_rl.misc import first_in_collection
from deep_rl.trajectories import discount
from six.moves import range


EPSILON_ENDS = [0.1, 0.01, 0.5]


class A3CAgent:
    """A3CAgent"""


    def __init__(self,
            graph,
            exploration_steps,
            total_steps,
            gamma,
            max_traj_len,
            action_sampler):
        """
        graph should have the placeholders called "states", "actions",
        and "returns". It should also have operations called "loss_op", "train_op",
        "probs", and "value".
        """

        self.graph = graph
        self.gamma = gamma
        self.max_traj_len = max_traj_len
        self.action_sampler = action_sampler

        self.T = graph.get_collection("global_step")[0]
        self.exploration_steps = exploration_steps
        self.total_steps = total_steps
        self.incr_T = tf.assign_add(self.T, 1)

    def pick_epsilon(self, T):
        rv = random.random()
        if rv < 0.4:
            end = EPSILON_ENDS[0]
        elif rv >= 0.4 and rv < 0.7:
            end = EPSILON_ENDS[1]
        else:
            end = EPSILON_ENDS[2]

        if T > self.exploration_steps:
            return end
        return 1.0 - T * (1.0 - end) / self.exploration_steps

    def run(self, t_id, session, coord, env):
        t = 0
        t_start = 0  # for updating params
        t_ep = 0  # for checking is an episode is done
        ep_reward = 0
        actions = []
        states = []
        rewards = []

        # inputs and ops
        _actions = self.graph.get_collection("actions")[0]
        _returns = self.graph.get_collection("returns")[0]
        _states = self.graph.get_collection("states")[0]
        train_op = self.graph.get_collection("train_op")[0]
        policy = self.graph.get_collection("policy")[0]
        value = self.graph.get_collection("value")[0]

        state = env.reset()

        try:
            while not coord.should_stop():
                T = session.run(self.T)
                if T > self.total_steps:
                    break

                epsilon = self.pick_epsilon(T)
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    # probs = session.run(policy,
                    #                     feed_dict={_states: state.reshape(1, *state.shape)})[0]
                    # action = self.action_sampler(probs)[0]
                    probs = session.run(policy, feed_dict={_states: state.reshape(1, *state.shape)})
                    action = self.action_sampler(probs)[0]

                next_state, reward, done, info = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                ep_reward += reward
                t += 1
                session.run(self.incr_T)

                # update params
                if done or t - t_start == self.max_traj_len:
                    last_state = states[-1]
                    val = 0
                    if not done:
                        val = session.run(value,
                                          feed_dict={_states:
                                                     last_state.reshape(1, *last_state.shape)})
                    rewards.append(val)
                    returns = discount(rewards, self.gamma)[:-1]

                    session.run([train_op],
                                feed_dict={_states: states,
                                           _returns: returns,
                                           _actions: actions})

                    actions = []
                    states = []
                    rewards = []
                    t_start = t

                # TODO: see if we can monitor all the envs
                if done or t - t_ep == env.spec.timestep_limit:
                    state = env.reset()
                    print("Thread id {}: Episode reward = {}, timestep = {}".format(t_id, ep_reward,
                                                                                    t))
                    ep_reward = 0
                    t_ep = t
                else:
                    state = next_state

        except Exception as e:
            coord.request_stop(e)

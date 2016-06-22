from __future__ import absolute_import, division, print_function

import random
import threading

import tensorflow as tf

import gym
from deep_rl.envs import EnvWrapper
from deep_rl.graphs import create_a3c_graph
from deep_rl.misc import first_in_collection
from deep_rl.trajectories import discount
from six.moves import range


# TODO: use tf.train.Supervisor
class A3CAgent:
    """A3CAgent"""

    def __init__(self,
                 env_name,
                 graph,
                 session,
                 gamma,
                 epsilon,
                 max_traj_len,
                 action_sampler,
                 outdir,
                 state_filter=None,
                 reward_filter=None,
                 num_threads=1):
        """
        graph should have the placeholders called "states", "actions",
        and "returns". It should also have operations called "loss_op", "train_op",
        "probs", and "value".
        """
        self._env_name = env_name
        self._g = graph
        self._s = session
        self._gamma = gamma
        self._epsilon = epsilon
        self._max_traj_len = max_traj_len
        self._action_sampler = action_sampler
        self._outdir = outdir
        self._reward_filter = reward_filter
        self._state_filter = state_filter
        self._num_threads = num_threads

    def _create_env(self, name):
        env = EnvWrapper(name, self._state_filter, self._reward_filter)
        return env

    def train(self):
        try:
            coord = tf.train.Coordinator()
            envs = [self._create_env(self._env_name)
                    for _ in range(self._num_threads)]
            # monitor the first env for future upload to scoreboard
            monitor_env = envs[0]
            monitor_env.monitor.start(self._outdir, force=True)
            threads = [threading.Thread(target=self.run_thread,
                                        args=(coord, envs[i]))
                       for i in range(self._num_threads)]
            for t in threads:
                t.start()
            coord.join(threads)
        except Exception as e:
            print("Error training in main loop ...")
            print(e)

    def run_thread(self, coord, env):
        t = 0
        t_start = 0
        actions = []
        states = []
        rewards = []

        # inputs and ops
        _actions = self._g.get_collection("actions")[0]
        _returns = self._g.get_collection("returns")[0]
        _states = self._g.get_collection("states")[0]
        train_op = self._g.get_collection("train_op")[0]
        policy = self._g.get_collection("policy")[0]
        value = self._g.get_collection("value")[0]
        global_step = self._g.get_collection("global_step")[0]

        state = env.reset()

        try:
            while not coord.should_stop():
                if random.random() < self._epsilon:
                    action = env.action_space.sample()
                else:
                    probs = self._s.run(
                        policy,
                        feed_dict={_states: state.reshape(1, -1)})[0]
                    action = self._action_sampler(probs)

                next_state, reward, done, info = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                t += 1
                # update params
                if done or t - t_start == self._max_traj_len:
                    last_state = states[-1]
                    val = 0
                    if not done:
                        val = self._s.run(
                            value,
                            feed_dict={_states: last_state.reshape(1, -1)})
                    rewards.append(val)
                    returns = discount(rewards, self._gamma)[:-1]

                    self._s.run(train_op,
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

        except Exception as e:
            coord.request_stop(e)

    def test(self, name=None, render=True):
        if not name: name = self._env_name
        env = self._create_env(name)
        policy = self._g.get_collection("policy")[0]
        _states = self._g.get_collection("states")[0]
        done = False
        total_reward = 0

        state = env.reset()
        while not done:
            probs = self._s.run(policy,
                                feed_dict={_states: state.reshape(1, -1)})[0]
            action = self._action_sampler(probs)
            state, reward, done, info = env.step(action)
            total_reward += reward
        env.close()

        return total_reward

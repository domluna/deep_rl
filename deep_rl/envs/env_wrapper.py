from __future__ import absolute_import, division, print_function

import gym


class EnvWrapper:
    """Wrapper around OpenAI Gym environments allowing
    for state and reward preprocessing."""

    def __init__(self, name, state_filter=None, reward_filter=None):
        self.name = name
        self.env = gym.make(name)
        self.state_filter = state_filter
        self.reward_filter = reward_filter
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        s, reward, done, info = self.env.step(action)
        s = self.state_filter(s) if self.state_filter else s
        reward = self.reward_filter(reward) if self.reward_filter else reward
        return s, reward, done, info

    def reset(self):
        s = self.env.reset()
        return self.state_filter(s) if self.state_filter else s

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

    def seed(self, seed=None):
        return self.env.seed(seed)

    def close(self):
        self.env.close()

    @property
    def monitor(self):
        return self.env.monitor

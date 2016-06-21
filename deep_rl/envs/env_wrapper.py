from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym


class EnvWrapper:
    """Wrapper around OpenAI Gym environments allowing
    for observation and reward preprocessing."""

    def __init__(self, name, ob_filter=None, reward_filter=None):
        self.name = name
        self.env = gym.make(name)
        self.ob_filter = ob_filter
        self.reward_filter = reward_filter
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = self.ob_filter(ob) if self.ob_filter else ob
        reward = self.reward_filter(reward) if self.reward_filter else reward
        return ob, reward, done, info

    def reset(self):
        ob = self.env.reset()
        return self.ob_filter(ob) if self.ob_filter else ob

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

    def seed(self, seed=None):
        return self.env.seed(seed)

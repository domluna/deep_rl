from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

class UniformExperienceReplay:
    """
    Store experiences for future lookup and training.

    Stored experiences will be sampled with a uniform distribution.
    """
    def __init__(self, capacity, batch_size, history_len, ob_shape, ob_dtype, flatten=False):
        self.capacity = capacity
        self.batch_size = batch_size
        self.history_len = history_len
        self.flatten = flatten
        self.index = 0
        self.size = 0

        obs_memory_shape = [capacity] + list(ob_shape)
        obs_batch_shape = [batch_size, history_len] + list(ob_shape)

        # memory
        self.obs = np.zeros(obs_memory_shape, dtype=ob_dtype)
        self.actions = np.zeros(capacity, dtype=np.uint8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.bool)

    def insert(self, ob, action, reward, terminal):
        """
        Inserts experience into replay.
        """
        self.obs[self.index, ...] = ob
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.terminals[self.index] = terminal

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self):
        end = self.size-self.history_len-1
        idxs = np.random.randint(0, end, self.batch_size)

        b_actions = self.actions[idxs]
        b_rewards =  self.rewards[idxs]
        b_terminals = self.terminals[idxs]

        b_obs = map(lambda i: self.obs[i:i+self.history_len], idxs)
        b_next_obs = map(lambda i: self.obs[i+1:i+1+self.history_len], idxs)
        b_obs = np.array(b_obs)
        b_next_obs = np.array(b_next_obs)

        if self.flatten:
            b_obs = b_obs.reshape(self.batch_size, -1)
            b_next_obs = b_next_obs.reshape(self.batch_size, -1)

        return b_obs, b_next_obs, b_actions, b_rewards, b_terminals

    def reset(self):
        self.obs[...] = 0
        self.actions[...] = 0
        self.rewards[...] = 0
        self.terminals[...] = 0

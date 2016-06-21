from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import deque
import numpy as np


class Buffer:
    """Rolling window of current state"""

    def __init__(self, capacity, flatten=False):
        self.capacity = capacity
        self.flatten = flatten
        self._state = deque([], capacity)

    def add(self, ob):
        """Shifts the observations to make room for the most recent one.
        The most recent observation should be on the last index"""
        self._state.append(ob)

    def reset(self):
        self._state.clear()

    @property
    def state(self):
        ret = np.array(self._state)
        if self.flatten:
            return ret.reshape(1, -1)
        return ret.reshape(1, *ret.shape)

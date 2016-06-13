from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from scipy.signal import lfilter
import numpy as np

def discount(x, gamma):
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_trajs(trajs, value_predict, gamma, gae_lambda):
    for t in trajs:
        t["returns"] = discount(t["rewards"], gamma)
        t["rewards_sum"] = t["rewards"].sum()
        v = t["baselines"] = value_predict(t["obs"])
        # we need V(s_t) and V(s_t+1), this makes the
        # later calculation easier
        v1 = np.append(v, 0 if t["terminated"] else v[-1])
        deltas = -v1[:-1] + t["rewards"] + gamma * v1[1:]
        t["advantages"] = discount(deltas, gamma * gae_lambda)
    all_advs = np.concatenate([t["advantages"] for t in trajs])
    mean = all_advs.mean()
    std = all_advs.std()
    for t in trajs:
        t["advantages"] = (t["advantages"] - mean) / std

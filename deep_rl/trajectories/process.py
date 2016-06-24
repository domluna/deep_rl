from __future__ import absolute_import, division, print_function

import numpy as np

from deep_rl.misc import discount


def compute_vpg_advantage(trajs, value_out, gamma, gae_lambda):
    for t in trajs:
        t["returns"] = discount(t["rewards"], gamma)
        t["rewards_sum"] = t["rewards"].sum()
        v = t["baselines"] = value_out(t["states"])
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

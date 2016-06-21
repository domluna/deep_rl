from __future__ import absolute_import, division, print_function

from collections import defaultdict

import numpy as np

from deep_rl.misc.distributions import categorical_sample
from six.moves import range

def sample_traj(env, compute_action, max_traj_len, render=False):
    traj = defaultdict(list)
    traj["terminated"] = False
    n_actions = env.action_space.n

    s = env.reset()
    for _ in range(max_traj_len):
        a = compute_action(s)
        next_s, r, done, _ = env.step(a)
        traj["states"].append(s)
        traj["actions"].append(a)
        traj["rewards"].append(r)
        s = next_s
        if done:
            traj["terminated"] = True
            break
        if render: env.render()

    return {k: np.array(v) for k, v in traj.items()}

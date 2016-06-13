from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import range
import numpy as np
from collections import defaultdict
from ..utils.distributions import categorical_sample

def sample_traj(env, policy_probs, max_traj_len, render=False):
    traj = defaultdict(list)
    traj["terminated"] = False
    n_actions = env.action_space.n

    ob = env.reset()
    for _ in range(max_traj_len):
        probs = policy_probs(ob)
        a = categorical_sample(probs)[0]
        next_ob, r, done, _ = env.step(a)
        traj["obs"].append(ob)
        traj["actions"].append(a)
        traj["rewards"].append(r)
        ob = next_ob
        if done:
            traj["terminated"] = True
            break
        if render: env.render()

    return {k:np.array(v) for k,v in traj.items()}

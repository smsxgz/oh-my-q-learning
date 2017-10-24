import numpy as np
from collections import namedtuple


def str_reward(rewards, k):
    """For print list of rewards."""
    output = '\n\n'
    output += '+' * 50 + '\n'
    output += '+  last one:\n'
    output += '+  {}\n'.format(rewards[-1])
    output += '+  recent {}:\n'.format(k)
    output += '+  max: {}, min: {}, mean: {:.3f}\n'.format(
        max(rewards[-k:]), min(rewards), np.mean(rewards[-k:]))
    output += '+  total:\n'
    output += '+  max: {}, mean: {:.3f}\n'.format(
        max(rewards), np.mean(rewards))
    output += '+' * 50 + '\n\n'

    return output


Transition = namedtuple("Transition",
                        ["state", "action", "reward", "next_state", "done"])

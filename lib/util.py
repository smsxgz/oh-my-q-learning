import numpy as np
from collections import namedtuple


def str_reward(rewards, k):
    """For print list of rewards."""
    output = [
        '+' * 50,
        'last one:',
        str(rewards[-1]),
        'recent {}:'.format(k),
        'max: {}, min: {}, mean: {:.3f}'.format(
            max(rewards[-k:]), min(rewards), np.mean(rewards[-k:])),
        'total:',
        'games: {}, max: {}, mean: {:.3f}'.format(
            len(rewards), max(rewards), np.mean(rewards)),
    ]

    return '\n+  '.join(output) + '\n' + '+' * 50 + '\n'


Transition = namedtuple("Transition",
                        ["state", "action", "reward", "next_state", "done"])

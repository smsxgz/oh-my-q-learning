import random
import numpy as np
import msgpack_numpy
from collections import deque
msgpack_numpy.patch()


class Memory(object):
    def __init__(self, capacity, init_size, batch_size, sample_every=1):
        self.mem = deque(maxlen=capacity)
        self.init_size = init_size
        self.batch_size = batch_size
        self.sample_every = sample_every
        self.tot = 0

    def append(self, transition):
        self.mem.append(transition)

    def extend(self, transitions):
        for t in transitions:
            self.append(t)

    def sample(self):
        if len(self.mem) <= self.init_size or self.tot == 0:
            return

        samples = random.sample(self.mem, self.batch_size)
        return map(np.array, zip(*samples))

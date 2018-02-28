import random
import numpy as np
import tensorflow as tf
from collections import deque
import os


def make_train_path():
    # make train dir
    cwd = os.getcwd()
    path = os.path.dirname(cwd)
    assert path[-6:] == 'config'

    basename = os.path.basename(cwd)
    true_train_path = os.path.join(path[:-6], 'train_log', basename)
    train_path = os.path.join(cwd, 'train_log')

    if not os.path.exists(true_train_path):
        os.makedirs(true_train_path)

    if not os.path.exists(train_path):
        os.system('ln -s {} train_log'.format(true_train_path))
    elif os.path.realpath(train_path) != true_train_path:
        os.system('rm train_log')
        os.system('ln -s {} train_log'.format(true_train_path))
    return train_path


train_path = make_train_path()


class EpsilonGreedy(object):
    def __init__(self,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=500000,
                 summary_writer=None):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.summary_writer = summary_writer

    def epsilon(self, global_step):
        rho = min(global_step,
                  self.epsilon_decay_steps) / self.epsilon_decay_steps
        return (1 - rho) * self.epsilon_start + rho * self.epsilon_end

    def __call__(self, q_values, global_step):
        epsilon = self.epsilon(global_step)
        if self.summary_writer:
            summary = tf.Summary()
            summary.value.add(simple_value=epsilon, tag='epsilon')
            self.summary_writer.add_summary(summary, global_step)

        batch_size = q_values.shape[0]
        best_actions = np.argmax(q_values, axis=1)
        actions = np.random.randint(0, q_values.shape[1], size=batch_size)
        idx = np.random.uniform(size=batch_size) > epsilon
        actions[idx] = best_actions[idx]
        return actions


class Memory(object):
    def __init__(self, capacity, batch_size):
        self.mem = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.tot = 0

    def append(self, transition):
        self.mem.append(transition)

    def extend(self, transitions):
        for t in transitions:
            self.append(t)

    def sample(self):
        samples = random.sample(self.mem, self.batch_size)
        self.tot += 1
        return map(np.array, zip(*samples))

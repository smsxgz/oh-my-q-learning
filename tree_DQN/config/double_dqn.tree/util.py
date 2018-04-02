import os
import random
import numpy as np
import torch
from torch.autograd import Variable
from collections import deque, defaultdict
from torch import nn


USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
INT = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    var = Variable(
        torch.from_numpy(ndarray),
        volatile=volatile,
        requires_grad=requires_grad).type(dtype)
    return var.cuda() if USE_CUDA else var


def make_train_path(train_prefix=None):
    # make train dir
    cwd = os.getcwd()
    path = os.path.dirname(cwd)
    assert path[-6:] == 'config'

    basename = os.path.basename(cwd)
    if train_prefix is not None:
        base_train_path = os.path.join(train_prefix)
        if not os.path.exists(base_train_path):
            os.makedirs(base_train_path)
        make_soft_link(base_train_path, os.path.join(path[:-6], 'train_log'))

    pre_train_path = os.path.join(path[:-6], 'train_log', basename)
    train_path = os.path.join(cwd, 'train_log')

    if not os.path.exists(pre_train_path):
        os.makedirs(pre_train_path)

    make_soft_link(pre_train_path, train_path)
    return train_path


def make_soft_link(base_path, path):
    try:
        os.system('rm {}'.format(path))
    except Exception:
        pass
    os.system('ln -s {} {}'.format(base_path, path))


train_path = make_train_path('/data1/ni_atari_train_logs')


class EpsilonGreedy(object):
    def __init__(self,
                 n_action,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=500000,
                 summary_writer=None):
        self.n_ac = n_action
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.summary_writer = summary_writer

    def epsilon(self, global_step):
        rho = min(global_step,
                  self.epsilon_decay_steps) / self.epsilon_decay_steps
        return (1 - rho) * self.epsilon_start + rho * self.epsilon_end

    def __call__(self, raw_actions, global_step):
        epsilon = self.epsilon(global_step)
        if global_step % 1000 == 0 and self.summary_writer:
            self.summary_writer.add_scalar('epsilon', epsilon, global_step)

        batch_size = raw_actions.shape
        actions = np.random.randint(self.n_ac, size=batch_size)
        idx = np.random.uniform(size=batch_size) > epsilon
        actions[idx] = raw_actions[idx]
        return actions


class Memory(object):
    def __init__(self, capacity):
        self.mem = deque(maxlen=capacity)

    def append(self, transition):
        self.mem.append(transition)

    def extend(self, transitions):
        for t in transitions:
            self.append(t)

    def sample(self, batch_size):
        samples = random.sample(self.mem, batch_size)
        samples = list(map(np.array, zip(*samples)))
        # state, action, reward, next_state, done
        samples[0] = samples[0].astype(np.float32) / 255.0
        samples[3] = samples[3].astype(np.float32) / 255.0
        return samples


class ResultsBuffer(object):
    def __init__(self, rewards_history=[]):
        self.buffer = defaultdict(list)

        assert isinstance(rewards_history, list)
        self.rewards_history = rewards_history

    def update_infos(self, info, total_t):
        for key in info:
            msg = info[key]
            self.buffer['reward'].append(msg[b'reward'])
            self.buffer['length'].append(msg[b'length'])
            if b'real_reward' in msg:
                self.buffer['real_reward'].append(msg[b'real_reward'])
                self.buffer['real_length'].append(msg[b'real_length'])
                self.rewards_history.append(
                    [total_t, key, msg[b'real_reward']])

    def update_summaries(self, summaries, total_t):
        for k in summaries:
            self.buffer[k].append(summaries[k])

    def add_summary(self, summary_writer, total_t, time):
        s = {'time': time}
        for key in self.buffer:
            if self.buffer[key]:
                s[key] = np.mean(self.buffer[key])
                self.buffer[key].clear()

        for key in s:
            summary_writer.add_scalar(key, s[key], total_t)
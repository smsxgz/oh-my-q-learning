import random
import numpy as np
from collections import deque
import torch


use_cuda = torch.cuda.is_available()


class EpsilonGreedy(object):
    def __init__(self,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=500000,
                 summary_writer=None):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = epsilon_decay_steps
        self.summary_writer = summary_writer

    def epsilon(self, global_step):
        rho = min(global_step, self.decay_steps) / self.decay_steps
        return (1 - rho) * self.epsilon_start + rho * self.epsilon_end

    def __call__(self, q_values, global_step):
        epsilon = self.epsilon(global_step)
        if global_step % 1000 == 0 and self.summary_writer:
            self.summary_writer.add_scalar('episode_info/epsilon', epsilon, global_step)

        batch_size = q_values.shape[0]
        best_actions = np.argmax(q_values, axis=1)
        actions = np.random.randint(0, q_values.shape[1], size=batch_size)
        idx = np.random.uniform(size=batch_size) > epsilon
        actions[idx] = best_actions[idx]
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
        return map(np.array, zip(*samples))


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    flat_params = torch.cat(params)
    return flat_params


def turn_into_cuda(var):
    return var.cuda() if use_cuda else var


def np_to_var(nparray):
    assert isinstance(nparray, np.ndarray)
    return torch.autograd.Variable(torch.from_numpy(nparray).float())
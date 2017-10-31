import random
import numpy as np
import tensorflow as tf


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
        self.tot = 0

    @property
    def epsilon(self):
        rho = min(self.tot,
                  self.epsilon_decay_steps) / self.epsilon_decay_steps
        return (1 - rho) * self.epsilon_start + rho * self.epsilon_end

    def __call__(self, q_values):
        epsilon = self.epsilon
        if self.summary_writer:
            summary = tf.Summary()
            summary.value.add(simple_value=epsilon, tag='epsilon')
            self.summary_writer.add_summary(summary, self.tot)

        if random.random() > epsilon:
            action = np.argmax(q_values)
        else:
            action = random.randint(0, len(q_values) - 1)
        self.tot += 1
        return action

import numpy as np
import tensorflow as tf
from .dqn import Dqn


class DoubleDqn(Dqn):
    def update(self, state_batch, action_batch, reward_batch, next_state_batch,
               done_batch):
        batch_size = state_batch.shape[0]
        next_q_vals, target_next_q_vals = self.sess.run(
            [self.qvals, self.target_qvals],
            feed_dict={self.input: next_state_batch})
        best_action = np.argmax(next_q_vals, axis=1)

        targets = reward_batch + (
            1 - done_batch) * self.discount * target_next_q_vals[np.arange(
                batch_size), best_action]
        _, total_t, loss, max_q_value = self.sess.run(
            [
                self.train_op,
                tf.train.get_global_step(), self.loss, self.max_qval
            ],
            feed_dict={
                self.input: state_batch,
                self.actions: action_batch,
                self.next_input: targets
            })
        return total_t, {'loss': loss, 'max_q_value': max_q_value}

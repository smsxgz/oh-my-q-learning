import numpy as np
import tensorflow as tf

from .tfestimator import TFEstimator


class Dqn(TFEstimator):
    def _build_model(self):
        # placeholders
        self.input = tf.placeholder(
            shape=[None, 84, 84, 4], dtype=tf.float32, name='inputs')
        self.actions = tf.placeholder(
            shape=[None], dtype=tf.int32, name='actions')
        self.next_input = tf.placeholder(
            shape=[None], dtype=tf.float32, name='next_inputs')

        # network
        with tf.variable_scope('qnet'):
            self.qvals = self._net(self.input)

        with tf.variable_scope('target'):
            self.target_qvals = self._net(self.input)

        batch_size = tf.shape(self.input)[0]
        gather_indices = tf.range(batch_size) * self.n_ac + self.actions
        self.action_q = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)
        self.loss = tf.reduce_mean(
            tf.squared_difference(self.next_input, self.action_q))
        self.max_qval = tf.reduce_max(self.qvals)

        self.train_op = self.optimizer.minimize(
            self.loss,
            global_step=tf.train.get_global_step(),
            var_list=tf.trainable_variables('qnet'))
        self.update_target_op = self._update_target_op()

    def update(self, state_batch, action_batch, reward_batch, next_state_batch,
               done_batch):
        batch_size = state_batch.shape[0]
        target_next_q_vals = self.sess.run(
            self.target_qvals, feed_dict={self.input: next_state_batch})
        best_action = np.argmax(target_next_q_vals, axis=1)

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


class DuelingNetwork(Dqn):
    def _net(self, x):
        conv1 = tf.contrib.layers.conv2d(x, 32, 8, 4)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1)
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        fc2 = tf.contrib.layers.fully_connected(
            fc1, self.n_ac + 1, activation_fn=None)
        out = fc2[:, 1:] + tf.expand_dims(
            fc2[:, 0] - tf.reduce_mean(fc2[:, 1:], axis=1), axis=1)
        return out

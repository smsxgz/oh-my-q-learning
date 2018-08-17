import numpy as np
import tensorflow as tf
from .tfestimator import TFEstimator


class DistributionalDqn(TFEstimator):
    def __init__(self,
                 n_ac,
                 lr=1e-4,
                 discount=0.99,
                 vmax=10,
                 vmin=-10,
                 n_atoms=51):
        self.n_atoms = n_atoms
        self.vmax = vmax
        self.vmin = vmin
        self.delta = (vmax - vmin) / (n_atoms - 1)
        self.split_points = np.linspace(vmin, vmax, n_atoms)
        super(DistributionalDqn, self).__init__(
            n_ac=n_ac, lr=lr, discount=discount)

    def _build_model(self):
        # placeholders
        self.input = tf.placeholder(
            shape=[None, 84, 84, 4], dtype=tf.float32, name='inputs')
        self.actions = tf.placeholder(
            shape=[None], dtype=tf.int32, name='actions')
        self.next_input = tf.placeholder(
            shape=[None, self.n_atoms], dtype=tf.float32, name='next_inputs')

        split_points = tf.constant(
            self.split_points, dtype=tf.float32, name='split_points')

        # network
        with tf.variable_scope('qnet'):
            self.logits = self._net(self.input)

        with tf.variable_scope('target'):
            self.target_logits = self._net(self.input)

        batch_size = tf.shape(self.input)[0]
        self.probs = tf.nn.softmax(
            tf.reshape(self.logits, [-1, self.n_ac, self.n_atoms]))
        self.probs_target = tf.nn.softmax(
            tf.reshape(self.target_logits, [-1, self.n_ac, self.n_atoms]))

        self.qvals = tf.reduce_sum(self.probs * split_points, axis=-1)
        self.max_qval = tf.reduce_max(self.qvals)

        gather_indices = tf.range(batch_size) * self.n_ac + self.actions
        self.action_probs = tf.gather(
            tf.reshape(self.probs, [-1, self.n_atoms]), gather_indices)
        self.action_probs_clip = tf.clip_by_value(self.action_probs, 0.00001,
                                                  0.99999)
        self.loss = tf.reduce_mean(-tf.reduce_sum(
            self.next_input * tf.log(self.action_probs_clip), axis=-1))
        self.train_op = self.optimizer.minimize(
            self.loss,
            global_step=tf.train.get_global_step(),
            var_list=tf.trainable_variables('qnet'))
        self.update_target_op = self._update_target_op()

    def _net(self, x, trainable=True):
        conv1 = tf.contrib.layers.conv2d(
            x, 32, 8, 4, activation_fn=tf.nn.relu, trainable=trainable)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu, trainable=trainable)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu, trainable=trainable)
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(
            flattened, 512, activation_fn=tf.nn.relu, trainable=trainable)
        fc2 = tf.contrib.layers.fully_connected(
            fc1,
            self.n_atoms * self.n_ac,
            activation_fn=None,
            trainable=trainable)
        return fc2

    def _calc_dist(self, reward, done, probs):
        discount = self.discount * (1 - done)
        m = np.zeros(self.n_atoms, dtype=np.float32)
        projections = (np.clip(reward + discount * self.split_points,
                               self.vmin, self.vmax) - self.vmin) / self.delta
        for (p, b) in zip(probs, projections):
            a = int(b)
            m[a] += p * (1 + a - b)
            if a < self.n_atoms - 1:
                m[a + 1] += p * (b - a)
        return m

    def update(self, state_batch, action_batch, reward_batch, next_state_batch,
               done_batch):
        batch_size = state_batch.shape[0]
        next_q_probs = self.sess.run(
            self.probs_target, feed_dict={self.input: next_state_batch})
        next_q_vals = np.sum(next_q_probs * self.split_points, axis=-1)
        best_action = np.argmax(next_q_vals, axis=1)

        targets = np.array([
            self._calc_dist(*args)
            for args in zip(reward_batch, done_batch, next_q_probs[np.arange(
                batch_size), best_action])
        ])

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

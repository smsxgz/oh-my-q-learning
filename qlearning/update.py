"""
    Manage q-estimator and target_estimator,
    especially for updating the q-estimator.
"""
import numpy as np
import tensorflow as tf
from qlearning.estimator import copy_model_parameters


class Update(object):
    def __init__(self,
                 sess,
                 checkpoint_dir,
                 q_estimator,
                 target_estimator,
                 discount_factor=0.99,
                 update_target_estimator_every=100,
                 save_model_every=25):

        self.q_estimator = q_estimator
        self.target_estimator = target_estimator
        self.sess = sess
        self.checkpoint_dir = checkpoint_dir
        self.discount_factor = discount_factor
        self.update_target_estimator_every = update_target_estimator_every
        self.save_model_every = save_model_every

        self.tot = 0
        self.saver = tf.train.Saver()

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)

    def __call__(self, *args):
        loss = self._update(*args)
        self.tot += 1
        print('{}th update, loss: {}'.format(self.tot, loss))

        if self.tot % self.update_target_estimator_every == 0:
            copy_model_parameters(self.sess, self.q_estimator,
                                  self.target_estimator)
            print("\nCopied model parameters to target network.")

        if self.tot % self.save_model_every == 0:
            self.saver.save(self.sess, self.checkpoint_dir + 'model')
            print("\nSave session.")

    def _update(self, states_batch, action_batch, reward_batch,
                next_states_batch, done_batch):
        pass


class DoubleDQNUpdate(Update):
    def __init__(self, **kwargs):
        super(DoubleDQNUpdate, self).__init__(**kwargs)

    def _update(self, states_batch, action_batch, reward_batch,
                next_states_batch, done_batch):
        batch_size = states_batch.shape[0]
        q_values_next = self.q_estimator.predict(self.sess, next_states_batch)
        best_actions = np.argmax(q_values_next, axis=1)
        q_values_next_target = self.target_estimator.predict(
            self.sess, next_states_batch)
        discount_factor = np.invert(done_batch).astype(
            np.float32) * self.discount_factor
        targets_batch = reward_batch + discount_factor * \
            q_values_next_target[np.arange(batch_size), best_actions]

        return self.q_estimator.update(self.sess, states_batch, action_batch,
                                       targets_batch)


class DistributionDQNUpdate(Update):
    def __init__(self, flag='naive', vmax=10, vmin=-10, N=51, **kwargs):
        super(DistributionDQNUpdate, self).__init__(**kwargs)
        self.N = N
        self.vmin = vmin
        self.vmax = vmax
        self.delta = (vmax - vmin) / (N - 1)
        self.split_points = np.linspace(vmin, vmax, N, dtype=np.float32)
        self.flag = flag

    def _intercept(self, x):
        if x > self.vmax:
            return self.vmax
        elif x < self.vmin:
            return self.vmin
        else:
            return x

    def _distribute(self, reward, discount_factor, probs):
        m = np.zeros(self.N, dtype=np.float32)
        for (p, z) in zip(probs, self.split_points):
            projection = self._intercept(reward + discount_factor * z)
            b = (projection - self.vmin) / self.delta
            l = int(b)
            m[l] += p * (1 + l - b)
            if l < self.N - 1:
                m[l + 1] += p * (b - l)
        return m

    def _update(self, *args):
        if self.flag == 'double':
            return self._double(*args)
        elif self.flag == 'simple':
            return self._simple(*args)
        elif self.flag == 'naive':
            return self._naive(*args)

    def _naive(self, states_batch, action_batch, reward_batch,
               next_states_batch, done_batch):
        """Don't need a target-estimator."""
        batch_size = states_batch.shape[0]
        q_probs_next, q_values_next = self.q_estimator.dis_predict(
            self.sess, next_states_batch)
        best_actions = np.argmax(q_values_next, axis=1)
        targets_batch = []
        for reward, probs, done in zip(
                reward_batch,
                q_probs_next[np.arange(batch_size), best_actions], done_batch):
            m = self._distribute(reward,
                                 self.discount_factor * (1 - float(done)),
                                 probs)
            targets_batch.append(m)
        targets_batch = np.array(targets_batch)

        return self.q_estimator.update(self.sess, states_batch, action_batch,
                                       targets_batch)

    def _simple(self, states_batch, action_batch, reward_batch,
                next_states_batch, done_batch):
        """Update q-estimator like DQN."""
        batch_size = states_batch.shape[0]
        q_probs_next, q_values_next = self.target_estimator.dis_predict(
            self.sess, next_states_batch)
        best_actions = np.argmax(q_values_next, axis=1)
        targets_batch = []
        for reward, probs, done in zip(
                reward_batch,
                q_probs_next[np.arange(batch_size), best_actions], done_batch):
            m = self._distribute(reward,
                                 self.discount_factor * (1 - float(done)),
                                 probs)
            targets_batch.append(m)
        targets_batch = np.array(targets_batch)

        return self.q_estimator.update(self.sess, states_batch, action_batch,
                                       targets_batch)

    def _double(self, states_batch, action_batch, reward_batch,
                next_states_batch, done_batch):
        """Update q-estimator like Double DQN."""
        batch_size = states_batch.shape[0]
        q_values_next = self.q_estimator.predict(self.sess, next_states_batch)
        best_actions = np.argmax(q_values_next, axis=1)
        q_probs_next_target, _ = self.target_estimator.dis_predict(
            self.sess, next_states_batch)

        targets_batch = []
        for reward, probs, done in zip(
                reward_batch,
                q_probs_next_target[np.arange(batch_size), best_actions],
                done_batch):
            m = self._distribute(reward,
                                 self.discount_factor * (1 - float(done)),
                                 probs)
            targets_batch.append(m)
        targets_batch = np.array(targets_batch)

        return self.q_estimator.update(self.sess, states_batch, action_batch,
                                       targets_batch)

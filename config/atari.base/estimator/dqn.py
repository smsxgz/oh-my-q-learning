import tensorflow as tf
from .tfestimator import TFEstimator


class Dqn(TFEstimator):
    def __init__(self, n_ac, lr=1e-4, discount=0.99):
        super(Dqn, self).__init__(n_ac, lr, discount)
        self.update_target()

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
            self.target_qvals = self._net(self.input, trainable=False)

        trainable_variables = tf.trainable_variables('qnet')
        batch_size = tf.shape(self.input)[0]
        gather_indices = tf.range(batch_size) * self.n_ac + self.actions
        action_q = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)
        self.loss = tf.reduce_mean(
            tf.squared_difference(self.next_input, action_q))
        self.max_qval = tf.reduce_max(self.qvals)

        self.train_op = self.optimizer.minimize(
            self.loss,
            global_step=tf.train.get_global_step(),
            var_list=trainable_variables)
        self.update_target_op = self._get_update_target_op()

    def _get_update_target_op(self):
        params1 = tf.trainable_variables('qnet')
        params1 = sorted(params1, key=lambda v: v.name)
        params2 = tf.global_variables('target')
        params2 = sorted(params2, key=lambda v: v.name)
        assert len(params1) == len(params2)

        update_ops = []
        for param1, param2 in zip(params1, params2):
            update_ops.append(param2.assign(param1))
        return update_ops

    def update(self, state_batch, action_batch, reward_batch, next_state_batch,
               done_batch):
        # batch_size = state_batch.shape[0]
        target_next_q_vals = self.sess.run(
            self.target_qvals, feed_dict={
                self.input: next_state_batch
            })

        targets = reward_batch + (
            1 - done_batch) * self.discount * target_next_q_vals.max(axis=1)
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

    def update_target(self):
        self.sess.run(self.update_target_op)

    def get_qvals(self, obs):
        return self.sess.run(self.qvals, feed_dict={self.input: obs})


class SoftDqn(TFEstimator):
    def __init__(self, n_ac, lr=1e-4, discount=0.99, tau=0.001):
        self.tau = tau
        TFEstimator.__init__(self, n_ac, lr, discount)

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
            self.target_qvals = self._net(self.input, trainable=False)

        trainable_variables = tf.trainable_variables('qnet')
        batch_size = tf.shape(self.input)[0]
        gather_indices = tf.range(batch_size) * self.n_ac + self.actions
        action_q = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)
        self.loss = tf.reduce_mean(
            tf.squared_difference(self.next_input, action_q))
        self.max_qval = tf.reduce_max(self.qvals)

        train_op = self.optimizer.minimize(
            self.loss,
            global_step=tf.train.get_global_step(),
            var_list=trainable_variables)

        with tf.control_dependencies([train_op]):
            self.update_target_op = self._get_update_target_op()

        self.train_op = tf.group(train_op, *self.update_target_op)

    def _get_update_target_op(self):
        params1 = tf.trainable_variables('qnet')
        params1 = sorted(params1, key=lambda v: v.name)
        params2 = tf.global_variables('target')
        params2 = sorted(params2, key=lambda v: v.name)
        assert len(params1) == len(params2)

        update_ops = []
        for param1, param2 in zip(params1, params2):
            update_ops.append(
                param2.assign(self.tau * (param1 - param2) + param2))
        return update_ops

    def update(self, state_batch, action_batch, reward_batch, next_state_batch,
               done_batch):
        # batch_size = state_batch.shape[0]
        target_next_q_vals = self.sess.run(
            self.target_qvals, feed_dict={
                self.input: next_state_batch
            })

        targets = reward_batch + (
            1 - done_batch) * self.discount * target_next_q_vals.max(axis=1)
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

    def update_target(self):
        pass

    def get_qvals(self, obs):
        return self.sess.run(self.qvals, feed_dict={self.input: obs})

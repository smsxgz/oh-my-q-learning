import numpy as np
import tensorflow as tf


class Doubledqn(object):
    def __init__(self, n_ac, discount=0.99, epsilon=0.05, lr=1e-4):
        tf.reset_default_graph()
        tf.Variable(0, name='global_step', trainable=False)
        self.n_ac = n_ac
        self.discount = discount
        self.epsilon = epsilon

        # placeholders
        self.input = tf.placeholder(
            shape=[None, 84, 84, 4], dtype=tf.float32, name='inputs')
        self.actions = tf.placeholder(
            shape=[None], dtype=tf.int32, name='actions')
        self.next_input = tf.placeholder(
            shape=[None], dtype=tf.float32, name='next_inputs')

        # network
        with tf.variable_scope('qnet'):
            self.qvals = self.net(self.input)
        with tf.variable_scope('target'):
            self.target_qvals = self.net(self.input)

        self.trainable_variables = tf.trainable_variables('qnet')
        batch_size = tf.shape(self.input)[0]
        gather_indices = tf.range(batch_size) * self.n_ac + self.actions
        self.action_q = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)
        self.loss = tf.reduce_mean(
            tf.squared_difference(self.next_input, self.action_q))
        self.max_qval = tf.reduce_max(self.qvals)

        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = self.optimizer.minimize(
            self.loss,
            global_step=tf.train.get_global_step(),
            var_list=self.trainable_variables)
        self.update_target_op = self._get_update_target_op()

        self.saver = tf.train.Saver(max_to_keep=5)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.update_target()

    def net(self, x):
        conv1 = tf.contrib.layers.conv2d(x, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(
            flattened, 512, activation_fn=tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(
            fc1, self.n_ac, activation_fn=None)
        return fc2

    def _get_update_target_op(self):
        params1 = tf.trainable_variables('qnet')
        params1 = sorted(params1, key=lambda v: v.name)
        params2 = tf.trainable_variables('target')
        params2 = sorted(params2, key=lambda v: v.name)

        update_ops = []
        for param1, param2 in zip(params1, params2):
            update_ops.append(param2.assign(param1))
        return update_ops

    def update_target(self):
        self.sess.run(self.update_target_op)

    def get_action(self, obs):
        qvals = self.sess.run(self.qvals, feed_dict={self.input: obs})
        best_action = np.argmax(qvals, axis=1)
        batch_size = obs.shape[0]
        actions = np.random.randint(self.n_ac, size=batch_size)
        idx = np.random.uniform(size=batch_size) > self.epsilon
        actions[idx] = best_action[idx]
        return actions

    def update(self, state_batch, action_batch, reward_batch, next_state_batch,
               done_batch):
        batch_size = state_batch.shape[0]
        next_q_vals, target_next_q_vals = self.sess.run(
            [self.qvals, self.target_qvals],
            feed_dict={
                self.input: next_state_batch
            })
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

    def save_model(self, outdir):
        total_t = self.sess.run(tf.train.get_global_step())
        self.saver.save(
            self.sess, outdir + '/model', total_t, write_meta_graph=False)

    def load_model(self, outdir):
        latest_checkpoint = tf.train.latest_checkpoint(outdir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            print("New start")

    def get_global_step(self):
        return self.sess.run(tf.train.get_global_step())

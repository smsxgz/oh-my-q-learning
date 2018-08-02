import numpy as np
import os
from collections import deque
import tensorflow as tf


class AAdoubledqn(object):
    def __init__(self, n_ac, k=2, discount=0.99, epsilon=0.05, lr=1e-4):
        tf.reset_default_graph()
        tf.Variable(0, name='global_step', trainable=False)
        self.n_ac = n_ac
        self.discount = discount
        self.epsilon = epsilon
        self.k = k
        # placeholders
        self.input = tf.placeholder(shape=[None, 4, 84, 84], dtype=tf.float32, name='inputs')
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32, name='actions')
        self.next_input = tf.placeholder(shape=[None, ], dtype=tf.float32, name='next_inputs')
        # network
        with tf.variable_scope('qnet'):
            self.qvals = self.net(tf.transpose(self.input, [0, 2, 3, 1]))

        self.targets_weights = np.array([1 / k for i in range(k)])
        self.targets = []
        for i in range(k):
            with tf.variable_scope('target' + str(i)):
                self.targets.append(self.net(tf.transpose(self.input, [0, 2, 3, 1])))

        self.trainable_variables = tf.trainable_variables('qnet')
        batch_size = tf.shape(self.input)[0]
        gather_indices = tf.range(batch_size) * self.n_ac + self.actions
        self.action_q = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)
        self.loss = tf.reduce_mean(tf.squared_difference(self.next_input, self.action_q))

        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step(),
                                                var_list=self.trainable_variables)
        self.update_target_op = self._update_target_op()

        self.saver = tf.train.Saver(max_to_keep=50)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        for i in range(self.k):
            self.update_target()

    def net(self, x):
        conv1 = tf.contrib.layers.conv2d(x, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(fc1, self.n_ac, activation_fn=None)
        return fc2

    def _update_target_op(self):
        params1 = tf.trainable_variables('qnet')
        params1 = sorted(params1, key=lambda v: v.name)
        all_params = []
        for i in range(self.k):
            params = tf.trainable_variables('target' + str(i))
            params = sorted(params, key=lambda v: v.name)
            all_params.append(params)
        all_params.append(params1)

        update_ops = []
        for i in range(self.k):
            for param1, param2 in zip(all_params[i + 1], all_params[i]):
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

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        batch_size = state_batch.shape[0]
        next_q_vals, target_next_q_vals = self.sess.run([self.qvals, self.targets],
                                                        feed_dict={self.input: next_state_batch})
        best_action = np.argmax(next_q_vals, axis=1)
        target_next_qvals_action = []
        for target_next_q_val in target_next_q_vals:
            target_next_qvals_action.append(target_next_q_val[np.arange(batch_size), best_action])
        target_next_qvals_action = np.array(target_next_qvals_action)

        targets = reward_batch + (1 - done_batch) * self.discount * np.dot(self.targets_weights,
                                                                           target_next_qvals_action)
        _, total_t, loss = self.sess.run([self.train_op, tf.train.get_global_step(), self.loss],
                                         feed_dict={self.input: state_batch, self.actions: action_batch,
                                                    self.next_input: targets})
        return total_t, {'loss': loss}

    def update_weights(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        batch_size = state_batch.shape[0]
        target_q_vals = self.sess.run(self.targets, feed_dict={self.input: state_batch})
        target_q_vals_action = [target_q_val[np.arange(batch_size), action_batch] for target_q_val in target_q_vals]

        next_q_vals, target_next_q_vals = self.sess.run([self.qvals, self.targets],
                                                        feed_dict={self.input: next_state_batch})
        best_action = np.argmax(next_q_vals, axis=1)
        errs = []
        for (target_next_q_val, target_q_val_action) in zip(target_next_q_vals, target_q_vals_action):
            errs.append(reward_batch + (1 - done_batch) * self.discount * target_next_q_val[
                np.arange(batch_size), best_action] - target_q_val_action)
        errs = np.array(errs)
        inv = np.linalg.inv(np.matmul(errs, errs.T))
        new_alphas = np.sum(inv, axis=1) / np.sum(inv)
        self.targets_weights = new_alphas

        return {'alpha_{}'.format(i + 1): self.targets_weights[i] for i in range(self.k)}

    def save_model(self, outdir):
        total_t = self.sess.run(tf.train.get_global_step())
        self.saver.save(self.sess, outdir + '/model', total_t, write_meta_graph=False)

    def load_model(self, outdir):
        latest_checkpoint = tf.train.latest_checkpoint(outdir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            print("New start")

    def get_global_step(self):
        return self.sess.run(tf.train.get_global_step())


class Distdqn(object):
    def __init__(self, n_ac, n_atoms, discount=0.99, epsilon=0.05, vmax=10, vmin=-10, lr=1e-4):
        tf.reset_default_graph()
        tf.Variable(0, name='global_step', trainable=False)
        self.n_ac = n_ac
        self.n_atoms = n_atoms
        self.discount = discount
        self.epsilon = epsilon
        self.vmax = vmax
        self.vmin = vmin
        self.delta = (vmax - vmin) / (n_atoms - 1)
        self.split_points = np.linspace(vmin, vmax, n_atoms)

        # placeholders
        self.input = tf.placeholder(shape=[None, 4, 84, 84], dtype=tf.float32, name='inputs')
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32, name='actions')
        self.next_input = tf.placeholder(shape=[None, self.n_atoms], dtype=tf.float32, name='next_inputs')
        # network
        with tf.variable_scope('distq'):
            self.logits = self.net(tf.transpose(self.input, [0, 2, 3, 1]))

        with tf.variable_scope('target'):
            self.target_logits = self.net(tf.transpose(self.input, [0, 2, 3, 1]))
        self.trainable_variables = tf.trainable_variables('distq')
        batch_size = tf.shape(self.input)[0]
        self.probs = tf.nn.softmax(tf.reshape(self.logits, [-1, self.n_ac, self.n_atoms]))
        self.probs_target = tf.nn.softmax(tf.reshape(self.target_logits, [-1, self.n_ac, self.n_atoms]))
        gather_indices = tf.range(batch_size) * self.n_ac + self.actions
        self.action_probs = tf.gather(tf.reshape(self.probs, [-1, self.n_atoms]), gather_indices)
        self.action_probs_clip = tf.clip_by_value(self.action_probs, 0.00001, 0.99999)
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.next_input * tf.log(self.action_probs_clip), axis=-1))
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step(),
                                                var_list=self.trainable_variables)
        self.update_target_op = self._update_target_op()

        self.saver = tf.train.Saver(max_to_keep=50)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.update_target()

    def net(self, x):
        conv1 = tf.contrib.layers.conv2d(x, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(fc1, self.n_atoms * self.n_ac, activation_fn=None)
        return fc2

    def _update_target_op(self):
        params1 = tf.trainable_variables('distq')
        params1 = sorted(params1, key=lambda v: v.name)
        params2 = tf.trainable_variables('target')
        params2 = sorted(params2, key=lambda v: v.name)

        update_ops = []
        for param1, param2 in zip(params1, params2):
            update_ops.append(param2.assign(param1))
        return update_ops

    def interception(self, x):
        if x > self.vmax:
            return self.vmax
        elif x < self.vmin:
            return self.vmin
        else:
            return x

    def calc_dist(self, reward, discount, probs):
        m = np.zeros(self.n_atoms, dtype=np.float32)
        for (p, z) in zip(probs, self.split_points):
            projection = self.interception(reward + discount * z)
            b = (projection - self.vmin) / self.delta
            l = int(b)
            m[l] += p * (1 + l - b)
            if l < self.n_atoms - 1:
                m[l + 1] += p * (b - l)
        return m

    def update_target(self):
        self.sess.run(self.update_target_op)

    def get_action(self, obs):
        probs = self.sess.run(self.probs, feed_dict={self.input: obs})
        qvals = np.sum(probs * self.split_points, axis=-1)
        best_action = np.argmax(qvals, axis=1)
        batch_size = obs.shape[0]
        actions = np.random.randint(self.n_ac, size=batch_size)
        idx = np.random.uniform(size=batch_size) > self.epsilon
        actions[idx] = best_action[idx]
        return actions

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        batch_size = state_batch.shape[0]
        next_q_probs = self.sess.run(self.probs_target, feed_dict={self.input: next_state_batch})
        next_q_vals = np.sum(next_q_probs * self.split_points, axis=-1)
        best_action = np.argmax(next_q_vals, axis=1)

        targets = []
        for reward, probs, done in zip(reward_batch, next_q_probs[np.arange(batch_size), best_action], done_batch):
            targets.append(self.calc_dist(reward, self.discount * (1 - done), probs))
        targets = np.array(targets)
        _, total_t, loss = self.sess.run([self.train_op, tf.train.get_global_step(), self.loss],
                                         feed_dict={self.input: state_batch, self.actions: action_batch,
                                                    self.next_input: targets})
        return total_t, {'loss': loss}

    def save_model(self, outdir):
        total_t = self.sess.run(tf.train.get_global_step())
        self.saver.save(self.sess, outdir + '/model', total_t, write_meta_graph=False)

    def load_model(self, outdir):
        latest_checkpoint = tf.train.latest_checkpoint(outdir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            print("New start")

    def get_global_step(self):
        return self.sess.run(tf.train.get_global_step())


class Doubledqn(object):
    def __init__(self, n_ac, discount=0.99, epsilon=0.05, lr=1e-4):
        tf.reset_default_graph()
        tf.Variable(0, name='global_step', trainable=False)
        self.n_ac = n_ac
        self.discount = discount
        self.epsilon = epsilon
        # placeholders
        self.input = tf.placeholder(shape=[None, 4, 84, 84], dtype=tf.float32, name='inputs')
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32, name='actions')
        self.next_input = tf.placeholder(shape=[None, ], dtype=tf.float32, name='next_inputs')
        # network
        with tf.variable_scope('qnet'):
            self.qvals = self.net(tf.transpose(self.input, [0, 2, 3, 1]))

        with tf.variable_scope('target'):
            self.target_qvals = self.net(tf.transpose(self.input, [0, 2, 3, 1]))

        self.trainable_variables = tf.trainable_variables('qnet')
        batch_size = tf.shape(self.input)[0]
        gather_indices = tf.range(batch_size) * self.n_ac + self.actions
        self.action_q = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)
        self.loss = tf.reduce_mean(tf.squared_difference(self.next_input, self.action_q))

        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step(),
                                                var_list=self.trainable_variables)
        self.update_target_op = self._update_target_op()

        self.saver = tf.train.Saver(max_to_keep=50)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.update_target()

    def net(self, x):
        conv1 = tf.contrib.layers.conv2d(x, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(fc1, self.n_ac, activation_fn=None)
        return fc2

    def _update_target_op(self):
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

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        batch_size = state_batch.shape[0]
        next_q_vals, target_next_q_vals = self.sess.run([self.qvals, self.target_qvals],
                                                        feed_dict={self.input: next_state_batch})
        best_action = np.argmax(next_q_vals, axis=1)

        targets = reward_batch + (1 - done_batch) * self.discount * target_next_q_vals[
            np.arange(batch_size), best_action]
        _, total_t, loss = self.sess.run([self.train_op, tf.train.get_global_step(), self.loss],
                                         feed_dict={self.input: state_batch, self.actions: action_batch,
                                                    self.next_input: targets})
        return total_t, {'loss': loss}

    def save_model(self, outdir):
        total_t = self.sess.run(tf.train.get_global_step())
        self.saver.save(self.sess, outdir + '/model', total_t, write_meta_graph=False)

    def load_model(self, outdir):
        latest_checkpoint = tf.train.latest_checkpoint(outdir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            print("New start")

    def get_global_step(self):
        return self.sess.run(tf.train.get_global_step())


class Dqn(object):
    def __init__(self, n_ac, discount=0.99, epsilon=0.05, lr=1e-4):
        tf.reset_default_graph()
        tf.Variable(0, name='global_step', trainable=False)
        self.n_ac = n_ac
        self.discount = discount
        self.epsilon = epsilon
        # placeholders
        self.input = tf.placeholder(shape=[None, 4, 84, 84], dtype=tf.float32, name='inputs')
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32, name='actions')
        self.next_input = tf.placeholder(shape=[None, ], dtype=tf.float32, name='next_inputs')
        # network
        with tf.variable_scope('qnet'):
            self.qvals = self.net(tf.transpose(self.input, [0, 2, 3, 1]))

        with tf.variable_scope('target'):
            self.target_qvals = self.net(tf.transpose(self.input, [0, 2, 3, 1]))

        self.trainable_variables = tf.trainable_variables('qnet')
        batch_size = tf.shape(self.input)[0]
        gather_indices = tf.range(batch_size) * self.n_ac + self.actions
        self.action_q = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)
        self.loss = tf.reduce_mean(tf.squared_difference(self.next_input, self.action_q))

        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step(),
                                                var_list=self.trainable_variables)
        self.update_target_op = self._update_target_op()

        self.saver = tf.train.Saver(max_to_keep=50)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.update_target()

    def net(self, x):
        conv1 = tf.contrib.layers.conv2d(x, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(fc1, self.n_ac, activation_fn=None)
        return fc2

    def _update_target_op(self):
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

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        batch_size = state_batch.shape[0]
        target_next_q_vals = self.sess.run(self.target_qvals, feed_dict={self.input: next_state_batch})
        best_action = np.argmax(target_next_q_vals, axis=1)

        targets = reward_batch + (1 - done_batch) * self.discount * target_next_q_vals[
            np.arange(batch_size), best_action]
        _, total_t, loss = self.sess.run([self.train_op, tf.train.get_global_step(), self.loss],
                                         feed_dict={self.input: state_batch, self.actions: action_batch,
                                                    self.next_input: targets})
        return total_t, {'loss': loss}

    def save_model(self, outdir):
        total_t = self.sess.run(tf.train.get_global_step())
        self.saver.save(self.sess, outdir + '/model', total_t, write_meta_graph=False)

    def load_model(self, outdir):
        latest_checkpoint = tf.train.latest_checkpoint(outdir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            print("New start")

    def get_global_step(self):
        return self.sess.run(tf.train.get_global_step())


class AAdqn(object):
    def __init__(self, n_ac, k=2, discount=0.99, epsilon=0.05, lr=1e-4):
        tf.reset_default_graph()
        tf.Variable(0, name='global_step', trainable=False)
        self.n_ac = n_ac
        self.discount = discount
        self.epsilon = epsilon
        self.k = k
        # placeholders
        self.input = tf.placeholder(shape=[None, 4, 84, 84], dtype=tf.float32, name='inputs')
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32, name='actions')
        self.next_input = tf.placeholder(shape=[None, ], dtype=tf.float32, name='next_inputs')
        # network
        with tf.variable_scope('qnet'):
            self.qvals = self.net(tf.transpose(self.input, [0, 2, 3, 1]))

        self.targets_weights = np.array([1 / k for i in range(k)])
        self.targets = []
        for i in range(k):
            with tf.variable_scope('target' + str(i)):
                self.targets.append(self.net(tf.transpose(self.input, [0, 2, 3, 1])))

        self.trainable_variables = tf.trainable_variables('qnet')
        batch_size = tf.shape(self.input)[0]
        gather_indices = tf.range(batch_size) * self.n_ac + self.actions
        self.action_q = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)
        self.loss = tf.reduce_mean(tf.squared_difference(self.next_input, self.action_q))

        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step(),
                                                var_list=self.trainable_variables)
        self.update_target_op = self._update_target_op()

        self.saver = tf.train.Saver(max_to_keep=50)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        for i in range(self.k):
            self.update_target()

    def net(self, x):
        conv1 = tf.contrib.layers.conv2d(x, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(fc1, self.n_ac, activation_fn=None)
        return fc2

    def _update_target_op(self):
        params1 = tf.trainable_variables('qnet')
        params1 = sorted(params1, key=lambda v: v.name)
        all_params = []
        for i in range(self.k):
            params = tf.trainable_variables('target' + str(i))
            params = sorted(params, key=lambda v: v.name)
            all_params.append(params)
        all_params.append(params1)

        update_ops = []
        for i in range(self.k):
            for param1, param2 in zip(all_params[i + 1], all_params[i]):
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

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        batch_size = state_batch.shape[0]
        target_next_q_vals = self.sess.run(self.targets, feed_dict={self.input: next_state_batch})
        best_actions = [np.argmax(target_next_q_val, axis=1) for target_next_q_val in target_next_q_vals]
        target_next_qvals_action = []
        for (target_next_q_val, best_action) in zip(target_next_q_vals, best_actions):
            target_next_qvals_action.append(target_next_q_val[np.arange(batch_size), best_action])
        target_next_qvals_action = np.array(target_next_qvals_action)

        targets = reward_batch + (1 - done_batch) * self.discount * np.dot(self.targets_weights,
                                                                           target_next_qvals_action)
        _, total_t, loss = self.sess.run([self.train_op, tf.train.get_global_step(), self.loss],
                                         feed_dict={self.input: state_batch, self.actions: action_batch,
                                                    self.next_input: targets})
        return total_t, {'loss': loss}

    def update_weights(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        batch_size = state_batch.shape[0]
        target_q_vals = self.sess.run(self.targets, feed_dict={self.input: state_batch})
        target_q_vals_action = [target_q_val[np.arange(batch_size), action_batch] for target_q_val in target_q_vals]

        target_next_q_vals = self.sess.run(self.targets, feed_dict={self.input: next_state_batch})
        best_actions = [np.argmax(target_next_q_val, axis=1) for target_next_q_val in target_next_q_vals]
        errs = []
        for (target_next_q_val, target_q_val_action, best_action) in zip(target_next_q_vals, target_q_vals_action, best_actions):
            errs.append(reward_batch + (1 - done_batch) * self.discount * target_next_q_val[
                np.arange(batch_size), best_action] - target_q_val_action)
        errs = np.array(errs)
        inv = np.linalg.inv(np.matmul(errs, errs.T))
        new_alphas = np.sum(inv, axis=1) / np.sum(inv)
        self.targets_weights = new_alphas

        return {'alpha_{}'.format(i + 1): self.targets_weights[i] for i in range(self.k)}

    def save_model(self, outdir):
        total_t = self.sess.run(tf.train.get_global_step())
        self.saver.save(self.sess, outdir + '/model', total_t, write_meta_graph=False)

    def load_model(self, outdir):
        latest_checkpoint = tf.train.latest_checkpoint(outdir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            print("New start")

    def get_global_step(self):
        return self.sess.run(tf.train.get_global_step())

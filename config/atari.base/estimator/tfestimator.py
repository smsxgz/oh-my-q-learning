import os
import numpy as np
import tensorflow as tf


class TFEstimator(object):
    def __init__(self, n_ac, lr=1e-4, discount=0.99):
        self.n_ac = n_ac
        self.discount = discount
        self.optimizer = tf.train.AdamOptimizer(lr, epsilon=1.5e-4)
        self._prepare()

    def _prepare(self):
        tf.reset_default_graph()
        tf.Variable(0, name='global_step', trainable=False)
        self.saver = tf.train.Saver(max_to_keep=5)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self._build_model()
        self.sess.run(tf.global_variables_initializer())

    def _build_model(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def update_target(self):
        raise NotImplementedError

    def get_qvals(self, obs):
        raise NotImplementedError

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
            fc1, self.n_ac, activation_fn=None, trainable=trainable)

        return fc2

    def get_action(self, obs, epsilon):
        qvals = self.get_qvals(obs)
        best_action = np.argmax(qvals, axis=1)
        batch_size = obs.shape[0]
        actions = np.random.randint(self.n_ac, size=batch_size)
        idx = np.random.uniform(size=batch_size) > epsilon
        actions[idx] = best_action[idx]
        return actions

    def save_model(self, outdir):
        total_t = self.sess.run(tf.train.get_global_step())
        self.saver.save(
            self.sess,
            os.path.join(outdir, 'model'),
            total_t,
            write_meta_graph=False)

    def load_model(self, outdir):
        latest_checkpoint = tf.train.latest_checkpoint(outdir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            print("New start!!")

    def get_global_step(self):
        return self.sess.run(tf.train.get_global_step())

import os
import numpy as np
import tensorflow as tf


class TFEstimator(object):
    def __init__(self, n_ac, lr=1e-4, discount=0.99):
        tf.reset_default_graph()
        tf.train.get_or_create_global_step()
        self.n_ac = n_ac
        self.discount = discount
        self.optimizer = tf.train.AdamOptimizer(lr, epsilon=1.5e-4)

        self._build_model()

        self.saver = tf.train.Saver(max_to_keep=5)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())
        self.update_target()

    def _build_model(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def update_target(self):
        self.sess.run(self.update_target_op)

    def get_qvals(self, obs):
        return self.sess.run(self.qvals, feed_dict={self.input: obs})

    def _net(self, x, trainable=True):
        conv1 = tf.contrib.layers.conv2d(x, 32, 8, 4)
        conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2)
        conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1)
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        fc2 = tf.contrib.layers.fully_connected(
            fc1, self.n_ac, activation_fn=None)
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

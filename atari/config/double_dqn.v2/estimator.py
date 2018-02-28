import tensorflow as tf


class Estimator(object):
    def __init__(
            self,
            state_shape,
            action_n,
            optimizer,
            update_target_rho=0.01,
    ):
        """
        Notes:
            if update_target_rho is 1, we will copy q's parameters to target's
            parameters, and we should set update_target_every to be larger
            like 1000.
        """
        self.optimizer = optimizer
        self.update_target_rho = update_target_rho
        self._build_model(state_shape, action_n)
        self.activation_fn = tf.nn.selu

    def _network(self, X, action_n):
        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=self.activation_fn)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=self.activation_fn)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=self.activation_fn)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(
            flattened, 512, activation_fn=self.activation_fn)
        fc = tf.contrib.layers.fully_connected(
            fc1, action_n, activation_fn=None)

        return fc

    def _get_update_target_op(self):
        e1_params = tf.trainable_variables('q')
        e1_params = sorted(e1_params, key=lambda v: v.name)

        e2_params = tf.trainable_variables('target')
        e2_params = sorted(e2_params, key=lambda v: v.name)

        assert len(e1_params) == len(e2_params)

        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign((e1_v - e2_v) * self.update_target_rho + e2_v)
            update_ops.append(op)
        return update_ops

    def _build_model(self, state_shape, action_n):
        """Builds the Tensorflow graph."""
        self.X_pl = tf.placeholder(
            shape=[None] + state_shape, dtype=tf.float32, name="X")
        self.actions_pl = tf.placeholder(
            shape=[None], dtype=tf.int32, name="actions")
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        with tf.variable_scope('q'):
            self.predictions = self._network(self.X_pl, action_n)
            self.vars = tf.trainable_variables('q')

            batch_size = tf.shape(self.X_pl)[0]

            # Get the predictions for the chosen actions only
            gather_indices = tf.range(batch_size) * tf.shape(
                self.predictions)[1] + self.actions_pl
            self.action_predictions = tf.gather(
                tf.reshape(self.predictions, [-1]), gather_indices)

            # Calcualte the loss
            self.losses = tf.squared_difference(self.y_pl,
                                                self.action_predictions)
            self.loss = tf.reduce_mean(self.losses)

            self.grads_and_vars = self.optimizer.compute_gradients(
                self.loss, self.vars)
            self.train_op = self.optimizer.apply_gradients(
                self.grads_and_vars, global_step=tf.train.get_global_step())

            self.summaries = tf.summary.merge([
                tf.summary.scalar("loss", self.loss),
                tf.summary.scalar("max_q_value",
                                  tf.reduce_max(self.predictions)),
                tf.summary.scalar("min_q_value",
                                  tf.reduce_min(self.predictions))
            ])

        with tf.variable_scope('target'):
            self.target_predictions = self._network(self.X_pl, action_n)
            self.update_target_op = self._get_update_target_op()

    def predict(self, sess, s):
        return sess.run(self.predictions, {self.X_pl: s})

    def target_predict(self, sess, s):
        return sess.run(self.target_predictions, {self.X_pl: s})

    def update(self, sess, s, a, y):
        feed_dict = {self.X_pl: s, self.actions_pl: a, self.y_pl: y}

        return sess.run([
            self.summaries,
            tf.train.get_global_step(), self.train_op, self.loss
        ], feed_dict)

    def target_update(self, sess):
        sess.run(self.update_target_op)

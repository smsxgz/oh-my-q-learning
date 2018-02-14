import tensorflow as tf


class Estimator(object):
    def __init__(
            self,
            state_n,
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
        self._build_model(state_n, action_n)

    @staticmethod
    def _network(X, action_n):
        output = tf.contrib.layers.fully_connected(X, 24)
        output = tf.contrib.layers.fully_connected(
            output, action_n, activation_fn=None)
        return output

    def _get_update_target_op(self):
        e2_params = [
            t for t in tf.trainable_variables() if t.name.startswith('target')
        ]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        e1_params = [
            t for t in tf.trainable_variables() if t.name.startswith('q')
        ]
        e1_params = sorted(e1_params, key=lambda v: v.name)

        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign((e1_v - e2_v) * self.update_target_rho + e2_v)
            update_ops.append(op)
        return update_ops

    def _build_model(self, state_n, action_n):
        """Builds the Tensorflow graph."""
        self.X_pl = tf.placeholder(
            shape=[None, state_n], dtype=tf.float32, name="X")
        self.actions_pl = tf.placeholder(
            shape=[None], dtype=tf.int32, name="actions")
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        with tf.variable_scope('q'):
            self.predictions = self._network(self.X_pl, action_n)
            
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

            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [[grad, var]
                                   for grad, var in self.grads_and_vars
                                   if grad is not None]
            self.train_op = self.optimizer.apply_gradients(
                self.grads_and_vars, global_step=tf.train.get_global_step())

            self.summaries = tf.summary.merge([
                tf.summary.scalar("loss", self.loss),
                # tf.summary.histogram("loss_hist", self.losses),
                # tf.summary.histogram("q_values_hist", self.predictions),
                tf.summary.scalar("max_q_value",
                                  tf.reduce_max(self.predictions)),
                tf.summary.scalar("min_q_value",
                                  tf.reduce_min(self.predictions))
            ])
        
        with tf.variable_scope('target'):
            self.target_predictions = self._network(self.X_pl, action_n)
            self.update_target_op = self._get_update_target_op()

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

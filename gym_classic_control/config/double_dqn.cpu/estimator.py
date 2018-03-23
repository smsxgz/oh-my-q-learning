import numpy as np
import tensorflow as tf


class Estimator(object):
    def __init__(
            self,
            state_shape,
            action_n,
            lr,
            update_target_rho=0.01,
    ):
        """
        Notes:
            if update_target_rho is 1, we will copy q's parameters to target's
            parameters, and we should set update_target_every to be larger
            like 1000.
        """
        tf.reset_default_graph()
        tf.Variable(0, name='global_step', trainable=False)

        self.activation_fn = tf.nn.relu
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.update_target_rho = update_target_rho
        self._build_model(state_shape, action_n)

        self.saver = tf.train.Saver(max_to_keep=5)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def _network(self, X, action_n):
        output = tf.contrib.layers.fully_connected(
            X, 24, activation_fn=tf.nn.selu)
        output = tf.contrib.layers.fully_connected(
            output, action_n, activation_fn=None)
        return output

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
                tf.summary.scalar("max_q_value", tf.reduce_max(
                    self.predictions)),
                tf.summary.scalar("min_q_value", tf.reduce_min(
                    self.predictions))
            ])

        with tf.variable_scope('target'):
            self.target_predictions = self._network(self.X_pl, action_n)
            self.update_target_op = self._get_update_target_op()

    def predict(self, s):
        return self.sess.run(self.predictions, {self.X_pl: s})

    def update(self, discount_factor, states_batch, action_batch, reward_batch,
               next_states_batch, done_batch):
        batch_size = states_batch.shape[0]

        q_values_next, q_values_next_target = self.sess.run(
            [self.predictions, self.target_predictions], {
                self.X_pl: next_states_batch
            })
        best_actions = np.argmax(q_values_next, axis=1)
        discount_factor_batch = np.invert(done_batch).astype(
            np.float32) * discount_factor
        targets_batch = reward_batch + discount_factor_batch * \
            q_values_next_target[np.arange(batch_size), best_actions]

        feed_dict = {
            self.X_pl: states_batch,
            self.actions_pl: action_batch,
            self.y_pl: targets_batch
        }

        _, total_t, *summaries = self.sess.run([
            self.train_op,
            tf.train.get_global_step(), self.loss, self.max_q_value,
            self.min_q_value
        ], feed_dict)
        return total_t, summaries

    def target_update(self):
        self.sess.run(self.update_target_op)

    def save(self, checkpoint_path):
        total_t = self.get_global_step()
        checkpoint = os.path.join(checkpoint_path, 'model')
        self.saver.save(self.sess, checkpoint, total_t, write_meta_graph=False)

    def restore(self, checkpoint_path):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint:
            print("Loading model checkpoint {}...".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            print('New start!!')

    def get_global_step(self):
        return self.sess.run(tf.train.get_global_step())

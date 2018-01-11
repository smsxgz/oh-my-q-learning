import tensorflow as tf


def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """

    e2_params = [
        t for t in tf.trainable_variables()
        if t.name.startswith(estimator2.scope)
    ]
    e2_params = sorted(e2_params, key=lambda v: v.name)
    name_list = [
        v.name.replace(estimator2.scope, estimator1.scope) for v in e2_params
    ]

    e1_params = [t for t in tf.trainable_variables() if t.name in name_list]
    e1_params = sorted(e1_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


class Estimator(object):
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self,
                 network,
                 optimizer=None,
                 summary_writer=None,
                 scope="estimator"):
        """
        Notes:
            network is callable,
            receives nothing, returns X's placeholder and last layer of model.
        """
        self.network = network
        self.scope = scope
        self.summary_writer = summary_writer
        self._build_model()

    def _build_model(self):
        """Builds the Tensorflow graph."""
        with tf.variable_scope(self.scope):
            self.actions_pl = tf.placeholder(
                shape=[None], dtype=tf.int32, name="actions")
            self.y_pl = tf.placeholder(
                shape=[None], dtype=tf.float32, name="y")

            self.X_pl, self.predictions = self.network(self.x_shape)

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

            if self.optimizer:
                self.train_op = self.optimizer.minimize(
                    self.loss,
                    global_step=tf.contrib.framework.get_global_step())

            if self.summary_writer:
                self.summaries = tf.summary.merge([
                    tf.summary.scalar("loss", self.loss),
                    tf.summary.histogram("loss_hist", self.losses),
                    tf.summary.histogram("q_values_hist", self.predictions),
                    tf.summary.scalar("max_q_value",
                                      tf.reduce_max(self.predictions))
                ])

    def predict(self, sess, s):
        return sess.run(self.predictions, {self.X_pl: s})

    def update(self, sess, s, a, y):
        feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a}
        summaries, global_step, _, loss = sess.run([
            self.summaries,
            tf.contrib.framework.get_global_step(), self.train_op, self.loss
        ], feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

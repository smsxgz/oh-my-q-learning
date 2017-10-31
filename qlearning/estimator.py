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
                 action_n,
                 optimizer=None,
                 summary_dir=None,
                 network=None,
                 x_shape=[None, 84, 84, 4],
                 scope="estimator"):
        """
        Args:
            action_n:    size of action space
            summary_dir: where to save summaries
            network:     a function of building graph before the last layer
            x_shape:     shape of X_placeholder, [?, 84, 84, 4] for image input
            scope:       name of the varible scope
        """
        self.action_n = action_n
        self.summary_dir = summary_dir
        self.network = network if network else self._network
        self.x_shape = x_shape
        self.scope = scope
        if summary_dir:
            summary_dir = summary_dir + self.scope
            self.summary_writer = tf.summary.FileWriter(summary_dir)
        else:
            self.summary_writer = None
        self._build_model()

    @staticmethod
    def _network(X_pl):
        x = tf.cast(X_pl, tf.float32) / 255.0
        conv1 = tf.contrib.layers.conv2d(x, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc = tf.contrib.layers.fully_connected(flattened, 512)

        return fc

    def _build_model(self):
        """Builds the Tensorflow graph."""
        with tf.variable_scope(self.scope):
            self.X_pl = tf.placeholder(
                shape=self.x_shape, dtype=tf.uint8, name="X")
            self.actions_pl = tf.placeholder(
                shape=[None], dtype=tf.int32, name="actions")
            self.y_pl = tf.placeholder(
                shape=[None], dtype=tf.float32, name="y")

            fc = self.network(self.X_pl)

            batch_size = tf.shape(self.X_pl)[0]
            self.predictions = tf.contrib.layers.fully_connected(
                fc, self.action_n, activation_fn=None)

            # Get the predictions for the chosen actions only
            gather_indices = tf.range(batch_size) * tf.shape(
                self.predictions)[1] + self.actions_pl
            self.action_predictions = tf.gather(
                tf.reshape(self.predictions, [-1]), gather_indices)

            # Calcualte the loss
            self.losses = tf.squared_difference(self.y_pl,
                                                self.action_predictions)
            self.loss = tf.reduce_mean(self.losses)

            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

            # Summaries for Tensorboard
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

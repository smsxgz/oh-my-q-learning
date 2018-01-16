import os
import sys
import numpy as np
import tensorflow as tf


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
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self._build_model()

    def _build_model(self):
        """Builds the Tensorflow graph."""
        with tf.variable_scope(self.scope):
            self.actions_pl = tf.placeholder(
                shape=[None], dtype=tf.int32, name="actions")
            self.y_pl = tf.placeholder(
                shape=[None], dtype=tf.float32, name="y")

            self.X_pl, self.predictions = self.network()

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
                    self.loss, global_step=tf.train.get_global_step())

            if self.summary_writer:
                self.summaries = tf.summary.merge([
                    tf.summary.scalar("loss", self.loss),
                    # tf.summary.histogram("loss_hist", self.losses),
                    # tf.summary.histogram("q_values_hist", self.predictions),
                    tf.summary.scalar("max_q_value",
                                      tf.reduce_max(self.predictions)),
                    tf.summary.scalar("min_q_value",
                                      tf.reduce_min(self.predictions))
                ])

    def predict(self, sess, s):
        return sess.run(self.predictions, {self.X_pl: s})

    def update(self, sess, s, a, y):
        feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a}
        summaries, global_step, _, loss = sess.run([
            self.summaries,
            tf.train.get_global_step(), self.train_op, self.loss
        ], feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss, global_step


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

class Update(object):
    def __init__(self,
                 sess,
                 checkpoint_path,
                 q_estimator,
                 target_estimator,
                 discount_factor=0.99,
                 update_target_estimator_every=100,
                 save_model_every=100):

        self.q_estimator = q_estimator
        self.target_estimator = target_estimator
        self.sess = sess
        self.checkpoint_path = checkpoint_path
        self.discount_factor = discount_factor
        self.update_target_estimator_every = update_target_estimator_every
        self.save_model_every = save_model_every

        self.tot = 0
        self.saver = tf.train.Saver()

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if latest_checkpoint:
            print("Loading model checkpoint {}...".format(latest_checkpoint))
            try:
                self.saver.restore(sess, latest_checkpoint)
            except Exception:
                print('Loading failed, we will Start from scratch!!')

    def __call__(self, *args):
        loss, global_step = self._update(*args)
        self.tot += 1
        print('\r{}th update, loss: {}'.format(global_step, loss), end='')
        sys.stdout.flush()
        
        if self.tot % self.update_target_estimator_every == 0:
            copy_model_parameters(self.sess, self.q_estimator,
                                  self.target_estimator)
            print("\nCopied model parameters to target network.")

        if self.tot % self.save_model_every == 0:
            self.saver.save(self.sess,
                            os.path.join(self.checkpoint_path, 'model'),
                            global_step)
            print("\nSave session.")

        return global_step

    def _update(self, states_batch, action_batch, reward_batch,
                next_states_batch, done_batch):
        batch_size = states_batch.shape[0]
        best_actions = np.argmax(
            self.q_estimator.predict(self.sess, next_states_batch), axis=1)
        q_values_next_target = self.target_estimator.predict(
            self.sess, next_states_batch)
        discount_factor = np.invert(done_batch).astype(
            np.float32) * self.discount_factor
        targets_batch = reward_batch + discount_factor * \
            q_values_next_target[np.arange(batch_size), best_actions]

        return self.q_estimator.update(self.sess, states_batch, action_batch,
                                       targets_batch)

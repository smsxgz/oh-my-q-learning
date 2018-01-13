import click
import numpy as np
import tensorflow as tf

from dqn import dqn
from agent import Agent
import common as settings
from util import Memory
from util import EpsilonGreedy
from estimator import Estimator
from estimator import Update


def get_network(state_n, action_n):
    def network():
        X = tf.placeholder(shape=[None, state_n], dtype=tf.float32, name="X")
        output = tf.contrib.layers.fully_connected(
            X, int(round(np.sqrt(state_n * action_n))))
        output = tf.contrib.layers.fully_connected(
            output, action_n, activation_fn=None)
        return X, output

    return network


@click.command()
@click.option('--game_name')
def main(game_name):
    env = Agent(8, game_name)

    tf.reset_default_graph()
    tf.Variable(0, name='global_step', trainable=False)
    summary_writer = tf.summary.FileWriter(settings.events_path)
    optimizer = tf.train.AdamOptimizer(1e-3)

    memory = Memory(500000, 50000, 64)
    policy_fn = EpsilonGreedy(1.0, 0.1, 50000, summary_writer)

    network = get_network(env.state_shape[0], env.action_n)
    q_estimator = Estimator(network, optimizer, summary_writer, "q")
    target_estimator = Estimator(network, scope="target_q")

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    update_fn = Update(
        sess,
        settings.models_path,
        q_estimator,
        target_estimator,
        discount_factor=0.99,
        update_target_estimator_every=1000,
        save_model_every=1000)

    dqn(sess, env, update_fn, q_estimator, memory, summary_writer, policy_fn)


if __name__ == '__main__':
    main()

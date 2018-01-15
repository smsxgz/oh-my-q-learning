import os
import click

used_gpu = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu
import tensorflow as tf

from dqn import dqn
from agent import Agent
from common import train_path
from util import Memory
from util import EpsilonGreedy
from estimator import Estimator


@click.command()
@click.option('--game_name', prompt='game name ')
# @click.option('--')
def main(game_name):
    env = Agent(64, game_name)

    events_path = os.path.join(train_path, game_name, 'events')
    if not os.path.exists(events_path):
        os.makedirs(events_path)

    models_path = os.path.join(train_path, game_name, 'models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    tf.reset_default_graph()
    tf.Variable(0, name='global_step', trainable=False)
    summary_writer = tf.summary.FileWriter(events_path)
    optimizer = tf.train.AdamOptimizer(1e-3)

    estimator = Estimator(
        env.state_shape[0], env.action_n, optimizer, update_target_rho=1)

    memory = Memory(500000, 50000, 128)
    policy_fn = EpsilonGreedy(0.5, 0.1, 25000, summary_writer)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    dqn(sess,
        env,
        estimator,
        memory,
        summary_writer,
        models_path,
        policy_fn,
        discount_factor=0.99,
        save_model_every=1000,
        update_target_every=1000,
        num_iterations=500000)


if __name__ == '__main__':
    main()

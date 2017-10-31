import os
import qlearning
import tensorflow as tf
from threading import Thread
from qlearning.agent import Agent
from train.parser import get_parser
from lib.policy import EpsilonGreedy
from qlearning.master import OffMaster
from zmq.eventloop.ioloop import IOLoop
from qlearning.master import estimator_worker


def train_atari():
    flags = get_parser()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.used_gpu

    tf.reset_default_graph()
    tf.Variable(0, name='global_step', trainable=False)
    q_estimator = qlearning.DistributionEstimator(
        action_n=flags.action_n,
        vmin=flags.vmin,
        vmax=flags.vmax,
        N=flags.N,
        summary_dir=flags.summary_dir,
        scope='q')

    target_estimator = qlearning.DistributionEstimator(
        action_n=flags.action_n,
        vmin=flags.vmin,
        vmax=flags.vmax,
        N=flags.N,
        scope='target_q')

    policy = EpsilonGreedy(
        epsilon_start=flags.epsilon_start,
        epsilon_end=flags.epsilon_end,
        epsilon_decay_steps=flags.epsilon_decay_steps,
        summary_writer=q_estimator.summary_writer)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        estimator_update_func = qlearning.DistributionDQNUpdate(
            flag='double',
            vmin=flags.vmin,
            vmax=flags.vmax,
            N=flags.N,
            sess=sess,
            checkpoint_dir=flags.path + '/checkpoints/',
            q_estimator=q_estimator,
            target_estimator=target_estimator,
            discount_factor=flags.discount_factor,
            update_target_estimator_every=flags.update_target_estimator_every,
            save_model_every=flags.save_model_every)

        for i in range(flags.num_worker):
            w = Thread(
                target=estimator_worker,
                args=(flags.url_worker, i, sess, q_estimator, policy))
            w.daemon = True
            w.start()

        for i in range(flags.num_agent):
            c = Agent(flags.game_name, flags.url_client, i)
            c.daemon = True
            c.start()

        OffMaster(
            init_memory_size=flags.init_memory_size,
            memory_size=flags.memory_size,
            estimator_update_every=flags.num_agent * 20,
            url_worker=flags.url_worker,
            url_client=flags.url_client,
            batch_size=flags.batch_size,
            estimator_update_callable=estimator_update_func)

        IOLoop.instance().start()


if __name__ == '__main__':
    train_atari()

import os
import click

used_gpu = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

from dqn import dqn
from util import EpsilonGreedy
from estimator import Estimator
from tensorboardX import SummaryWriter
from wrapper import atari_env


def main(game_name):
    summary_writer = SummaryWriter('./runs/' + game_name)

    policy_fn = EpsilonGreedy(0.5, 0.01, 625000, summary_writer)

    env = atari_env(game_name)
    estimator = Estimator(env.action_space.n, 1e-4, update_target_rho=1.0)

    dqn(env,
        estimator,
        32,
        summary_writer,
        policy_fn,
        discount_factor=0.99,
        update_target_every=1000,
        learning_starts=200,
        memory_size=100000,
        num_iterations=6250000,
        update_every=4)


if __name__ == '__main__':
    main('PongNoFrameskip-v4')

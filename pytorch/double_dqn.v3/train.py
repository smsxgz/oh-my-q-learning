import os

used_gpu = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

from dqn import dqn
from util import EpsilonGreedy
from estimator import Estimator
from tensorboardX import SummaryWriter
from wrapper import atari_env
import argparse


def main(env_name,
         init_epsilon,
         final_epsilon,
         epsilon_decay_len,
         lr,
         rho,
         batch_size,
         gamma,
         update_target_every,
         warm_up,
         memory_size,
         num_iterations,
         update_every):
    summary_writer = SummaryWriter('./runs/' + env_name + '_' + str(update_every))
    policy_fn = EpsilonGreedy(init_epsilon, final_epsilon, epsilon_decay_len, summary_writer)
    env = atari_env(env_name)
    estimator = Estimator(env.action_space.n, lr, update_target_rho=rho)

    dqn(env,
        estimator,
        batch_size=batch_size,
        summary_writer=summary_writer,
        exploration_policy_fn=policy_fn,
        gamma=gamma,
        update_target_every=update_target_every,
        warm_up=warm_up,
        memory_size=memory_size,
        num_iterations=num_iterations,
        update_every=update_every)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--init_epsilon', type=float, default=0.5)
    parser.add_argument('--final_epsilon', type=float, default=0.01)
    parser.add_argument('--epsilon_decay_len', type=int, default=625000)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--rho', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--update_target_every', type=int, default=1000)
    parser.add_argument('--warm_up', type=int, default=200)
    parser.add_argument('--memory_size', type=int, default=100000)
    parser.add_argument('--num_iterations', type=int, default=6250000)
    parser.add_argument('--update_every', type=int, default=4)

    args = parser.parse_args()
    args = vars(args)

    for k in args:
        print(k, ':', args[k])
    main(**args)

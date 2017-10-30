import os
import gym
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--game_name', type=str)
    parser.add_argument('-p', '--path', type=str)
    parser.add_argument('--used_gpu', type=str, default='0')
    parser.add_argument('--num_agent', type=int, default=8)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=256)

    # for epsilon greedy policy
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.04)
    parser.add_argument('--epsilon_decay_steps', type=int, default=500000)

    # for update functions
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument(
        '--update_target_estimator_every', type=int, default=1000)
    parser.add_argument('--save_model_every', type=int, default=100)

    # for memory buffer
    parser.add_argument('--init_memory_size', type=int, default=50000)
    parser.add_argument('--memory_size', type=int, default=500000)

    # for distributional DQN
    parser.add_argument('--vmin', type=float, default=-10.0)
    parser.add_argument('--vmax', type=float, default=10.0)
    parser.add_argument('--N', type=int, default=51)

    flags = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.used_gpu
    flags.path = './experiment/' + flags.path + '/' + flags.game_name[:-3]
    flags.summary_dir = flags.path + '/summaries/'
    flags.url_worker = 'ipc://./tmp/{}-backend.ipc'.format(
        flags.game_name[:-3])
    flags.url_client = 'ipc://./tmp/{}-frontend.ipc'.format(
        flags.game_name[:-3])

    env = gym.make(flags.game_name)
    flags.action_n = env.action_space.n
    env.close()
    return flags

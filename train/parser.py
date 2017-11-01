import os
import argparse
import train as FLAGS


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--game_name', type=str)
    parser.add_argument('-p', '--path', type=str, default=FLAGS.path)
    parser.add_argument('--used_gpu', type=str, default=FLAGS.used_gpu)
    parser.add_argument('--num_agent', type=int, default=FLAGS.num_agent)
    parser.add_argument('--num_worker', type=int, default=FLAGS.num_worker)
    parser.add_argument('--batch_size', type=int, default=FLAGS.batch_size)
    parser.add_argument(
        '--learning_rate', type=float, default=FLAGS.learning_rate)

    # for epsilon greedy policy
    parser.add_argument(
        '--epsilon_start', type=float, default=FLAGS.epsilon_start)
    parser.add_argument('--epsilon_end', type=float, default=FLAGS.epsilon_end)
    parser.add_argument(
        '--epsilon_decay_steps', type=int, default=FLAGS.epsilon_decay_steps)

    # for update functions
    parser.add_argument(
        '--discount_factor', type=float, default=FLAGS.discount_factor)
    parser.add_argument(
        '--update_target_estimator_every',
        type=int,
        default=FLAGS.update_target_estimator_every)
    parser.add_argument(
        '--save_model_every', type=int, default=FLAGS.save_model_every)
    parser.add_argument(
        '--save_rewards_every', type=int, default=FLAGS.save_rewards_every)

    # for memory buffer
    parser.add_argument(
        '--update_estimator_every',
        type=int,
        default=FLAGS.update_estimator_every)
    parser.add_argument(
        '--init_memory_size', type=int, default=FLAGS.init_memory_size)
    parser.add_argument('--memory_size', type=int, default=FLAGS.memory_size)

    # for distributional DQN
    parser.add_argument('--vmin', type=float, default=FLAGS.vmin)
    parser.add_argument('--vmax', type=float, default=FLAGS.vmax)
    parser.add_argument('--N', type=int, default=FLAGS.N)

    flags = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.used_gpu
    flags.path = './experiment/' + flags.path + '/' + flags.game_name[:-3]
    flags.summary_dir = flags.path + '/summaries/'
    flags.url_worker = 'ipc://./tmp/{}-backend.ipc'.format(
        flags.game_name[:-3])
    flags.url_client = 'ipc://./tmp/{}-frontend.ipc'.format(
        flags.game_name[:-3])

    flags.network_struct = FLAGS.network_struct

    return flags

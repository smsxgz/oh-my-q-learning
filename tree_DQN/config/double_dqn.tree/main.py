import os
import click
import traceback
from tensorboardX import SummaryWriter

from train import train
from agent import Agent
from util import train_path
from util import EpsilonGreedy
from doubledqn import DoubleDQN


@click.command()
@click.option('--game_name', prompt='game name ')
@click.option('--basename', default=None)
@click.option('--lr', default=6.25e-5)
def main(game_name, basename, lr):
    assert 'NoFrameskip-v4' in game_name

    if basename is None:
        basename = game_name[:-14]
    else:
        basename = '{}-{}'.format(game_name[:-14], basename)

    events_path = os.path.join(train_path, basename, 'events')
    if not os.path.exists(events_path):
        os.makedirs(events_path)

    models_path = os.path.join(train_path, basename, 'models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    summary_writer = SummaryWriter(events_path)

    env = Agent(32, game_name, basename)
    estimator = DoubleDQN(env.state_shape, env.action_n, lr, update_target_rho=1.0)
    # estimator = DistDQN(env.state_shape, env.action_n, lr, update_target_rho=1.0, n_atoms=51)
    policy_fn = EpsilonGreedy(env.action_n, 0.5, 0.01, 625000, summary_writer)

    try:
        train(env,
              basename,
              estimator,
              32,
              summary_writer,
              models_path,
              policy_fn,
              gamma=0.99,
              update_target_every=1000,
              learning_starts=200,
              memory_size=100000,
              num_iterations=6250000)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt!!")
    except Exception:
        traceback.print_exc()
    finally:
        env.close()


if __name__ == '__main__':
    main()

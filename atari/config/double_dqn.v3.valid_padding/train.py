import os
import click
import traceback
from tensorboardX import SummaryWriter

from dqn import dqn
from agent import Agent
from util import train_path
from util import EpsilonGreedy
from estimator import Estimator


@click.command()
@click.option('--game_name', prompt='game name ')
def main(game_name):
    events_path = os.path.join(train_path, game_name, 'events')
    if not os.path.exists(events_path):
        os.makedirs(events_path)

    models_path = os.path.join(train_path, game_name, 'models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    summary_writer = SummaryWriter(events_path)

    policy_fn = EpsilonGreedy(0.5, 0.01, 625000, summary_writer)

    env = Agent(32, game_name)
    estimator = Estimator(
        env.state_shape, env.action_n, 1e-4, update_target_rho=1.0)

    try:
        dqn(env,
            estimator,
            32,
            summary_writer,
            models_path,
            policy_fn,
            discount_factor=0.99,
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

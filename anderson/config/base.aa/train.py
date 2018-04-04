import os
import click
import traceback
from tensorboardX import SummaryWriter

from dqn import dqn
from agent import Agent
from util import train_path
from util import EpsilonGreedy
from estimator import AA_Estimator, Estimator


@click.command()
@click.option('--game_name', prompt='game name ')
@click.option('--basename', default=None)
@click.option('--lr', type=float, default=6.25e-5)
@click.option('--update_target_every', type=int, default=1000)
@click.option('--anderson', type=bool, default=True)
def main(game_name, basename, lr, update_target_every, anderson):
    assert 'NoFrameskip-v4' in game_name

    if basename is None:
        basename = game_name[:-14]
    else:
        basename = '{}-{}'.format(game_name[:-14], basename)
    if anderson:
        basename = '{}-{}'.format(basename, 'anderson')

    events_path = os.path.join(train_path, basename, 'events')
    if not os.path.exists(events_path):
        os.makedirs(events_path)

    models_path = os.path.join(train_path, basename, 'models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    summary_writer = SummaryWriter(events_path)
    policy_fn = EpsilonGreedy(0.5, 0.01, 625000, summary_writer)

    env = Agent(32, game_name, basename)
    estimator = AA_Estimator(env.state_shape, env.action_n, lr) if anderson else Estimator(env.state_shape,
                                                                                           env.action_n, lr)

    try:
        dqn(env=env,
            basename=basename,
            estimator=estimator,
            batch_size=32,
            alpha_batch_size=32 * 10,
            summary_writer=summary_writer,
            checkpoint_path=models_path,
            exploration_policy_fn=policy_fn,
            discount_factor=0.99,
            update_target_every=update_target_every,
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

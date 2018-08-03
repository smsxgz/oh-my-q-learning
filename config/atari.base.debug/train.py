import os
import click
import traceback
from dqn import dqn
from agent import Agent
from estimator import get_estimator
from util import train_path


@click.command()
@click.option('--game_name', prompt='game name:')
@click.option('--lr', type=float, default=0.0001)
@click.option('--num_agents', type=int, default=32)
@click.option('--update_target_every', type=int, default=1000)
@click.option('--model_name', default='dqn')
def main(game_name, lr, num_agents, update_target_every, model_name):
    assert 'NoFrameskip-v4' in game_name

    basename = '{}:lr={}:na={}:ute={}:{}'.format(
        game_name[:-14], lr, num_agents, update_target_every, model_name)

    env = Agent(num_agents, game_name, basename)
    try:
        estimator = get_estimator(model_name, env.action_n, lr, 0.99)
        base_path = os.path.join(train_path, basename)
        print("start training!!")
        dqn(env,
            estimator,
            base_path,
            batch_size=32,
            epsilon=0.01,
            save_model_every=1000,
            update_target_every=update_target_every,
            learning_starts=200,
            memory_size=100000,
            num_iterations=40000000)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt!!")
    except Exception:
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()

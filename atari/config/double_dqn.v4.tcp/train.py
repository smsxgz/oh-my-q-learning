import os
import click
import traceback
from dqn import dqn
from agent import Agent
from util import train_path
from model import Doubledqn


@click.command()
@click.option('--game_name', prompt='game name:')
@click.option('--lr', type=float, default=0.0000625)
@click.option('--basename', default='')
@click.option('--port', type=int, default=7878)
@click.option('--num_agents', type=int, default=32)
def main(game_name, basename, lr, port, num_agents):
    assert 'NoFrameskip-v4' in game_name

    tmp = [game_name[:-14], str(lr)]
    if basename:
        tmp.append(basename)
    basename = ':'.join(tmp)
    base_path = os.path.join(train_path, basename)

    env = Agent(num_agents, game_name, port)
    model = Doubledqn(env.action_n, epsilon=0.05, lr=lr)
    try:
        print("start training!!")
        dqn(env,
            model,
            base_path,
            batch_size=32,
            save_model_every=1000,
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


if __name__ == "__main__":
    main()

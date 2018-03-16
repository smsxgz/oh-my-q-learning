import os
import click
import traceback
from tensorboardX import SummaryWriter

from distdqn import distdqn
from agent import Agent
from util import train_path
from model import Distdqn

@click.command()
@click.option('--game_name', prompt='game name:')
@click.option('--lr', default=0.00025)
@click.option('--basename', default='')
def main(game_name, basename, lr):
    events_path = os.path.join(train_path, game_name+':'+str(lr)+':'+basename, 'events')
    if not os.path.exists(events_path):
        os.makedirs(events_path)

    models_path = os.path.join(train_path, game_name+':'+str(lr)+':'+basename, 'models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    summary_writer = SummaryWriter(events_path)

    env = Agent(32, game_name)
    model = Distdqn(env.action_n, 51, epsilon=0.01, lr=lr, use_cuda=True)
    try:
        print("start training!")
        distdqn(env, model, 32, summary_writer, models_path, save_model_every=1000, update_target_every=10000, learning_starts=200, memory_size=100000, num_iterations=6250000)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt!!")
    except Exception:
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    main()

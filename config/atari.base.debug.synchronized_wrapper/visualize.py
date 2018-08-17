import os
import time
import click
import imageio
import numpy as np
from tqdm import tqdm

from wrapper import atari_env
from estimator import get_estimator
train_path = './train_log'

@click.command()
@click.option('--game_name')
@click.option('--model_name', default='dqn')
@click.option('--write_video', is_flag=True)
def main(game_name, model_name, write_video):
    assert 'NoFrameskip-v4' in game_name
    env = atari_env(game_name)

    estimator = get_estimator(model_name, env.action_space.n, 0.001, 0.99)

    basename_list = [
        name for name in os.listdir(train_path) if (game_name[:-14] in name) and (model_name in name)
    ]
    print(basename_list)

    def visualize(basename):
        checkpoint_path = os.path.join(train_path, basename, 'models')
        estimator.load_model(checkpoint_path)

        total_t = estimator.get_global_step()
        if not os.path.exists('./videos'):
            os.makedirs('./videos')
        videoWriter = imageio.get_writer(
            './videos/{}-{}.mp4'.format(basename, total_t), fps=30)

        state = env.reset(videowriter=videoWriter)
        lives = env.unwrapped.ale.lives()
        print(lives)
        r = 0
        tot = 0
        while True:
            action = estimator.get_action(np.array([state]), 0.01)
            state, reward, done, info = env.step(action)
            r += reward
            tot += 1
            if done:
                lives = env.unwrapped.ale.lives()
                print(lives)
                if info['was_real_done']:
                    print(tot, r)
                    break
                else:
                    state = env.reset()
        videoWriter.close()

    def evaluate(basename, num_eval=10):
        checkpoint_path = os.path.join(train_path, basename, 'models')
        estimator.load_model(checkpoint_path)

        res = []
        for i in tqdm(range(num_eval)):
            env.seed(int(time.time() * 1000) // 2147483647)
            state = env.reset()
            r = 0
            while True:
                action = estimator.get_action(np.array([state]), 0.01)
                state, reward, done, info = env.step(action)
                r += reward
                if done:
                    if info['was_real_done']:
                        res.append(r)
                        break
                    else:
                        state = env.reset()
        print('mean: {}, max: {}'.format(sum(res) / num_eval, max(res)))

    if write_video:
        for basename in basename_list:
            print("Writing {}'s video ...".format(basename))
            visualize(basename)
    else:
        for basename in basename_list:
            print("Evaluating {} ...".format(basename))
            evaluate(basename)


if __name__ == '__main__':
    main()

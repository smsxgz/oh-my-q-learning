import os
import time
import click
import imageio
import numpy as np
from tqdm import tqdm

from util import train_path
from wrapper import atari_env
from estimator import Estimator


@click.command()
@click.option('--game_name')
@click.option('--write_video', is_flag=True)
def main(game_name, write_video):
    assert 'NoFrameskip-v4' in game_name
    env = atari_env(game_name)

    estimator = Estimator(
        env.observation_space.shape,
        env.action_space.n,
        1e-4)

    basename_list = [
        name for name in os.listdir(train_path) if game_name[:-14] in name
    ]
    print(basename_list)

    def visualize(basename):
        checkpoint_path = os.path.join(train_path, basename, 'models')
        total_t = estimator.restore(checkpoint_path)

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
            q_value = estimator.predict(np.array([state]))
            action = np.argmax(q_value[0])
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
        estimator.restore(checkpoint_path)

        res = []
        for i in tqdm(range(num_eval)):
            env.seed(int(time.time() * 1000) // 2147483647)
            state = env.reset()
            r = 0
            while True:
                q_value = estimator.predict(np.array([state]))
                action = np.argmax(q_value[0])
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

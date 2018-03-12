import os
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
    env = atari_env(game_name)

    estimator = Estimator(
        env.observation_space.shape,
        env.action_space.n,
        1e-4,
        update_target_rho=1.0)

    checkpoint_path = os.path.join(train_path, game_name, 'models')
    estimator.restore(checkpoint_path)

    def visualize():
        total_t = estimator.get_global_step()
        videoWriter = imageio.get_writer(
            'train_log/{}-{}.mp4'.format(game_name, total_t), fps=30)

        state = env.reset(videowriter=videoWriter)
        lives = env.unwrapped.ale.lives()
        print(lives)
        r = 0
        tot = 0
        while True:
            q_value = estimator.predict([state])
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

    def evaluate():
        res = []
        for i in tqdm(range(50)):
            state = env.reset()
            r = 0
            while True:
                q_value = estimator.predict([state])
                action = np.argmax(q_value[0])
                state, reward, done, info = env.step(action)
                r += reward
                if done:
                    if info['was_real_done']:
                        res.append(r)
                        break
                    else:
                        state = env.reset()
        print(sum(res) / 50)

    if write_video:
        print('Writing video ...')
        visualize()
    else:
        print('Evaluating ...')
        evaluate()


if __name__ == '__main__':
    main()

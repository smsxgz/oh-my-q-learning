import os
import gym
import click
import imageio
import numpy as np

from common import train_path
from estimator import Estimator


@click.command()
@click.option('--game_name')
def main(game_name):
    env = gym.make(game_name)

    estimator = Estimator(
        env.observation_space.shape[0],
        env.action_space.n,
        1e-3,
        update_target_rho=1.0)

    checkpoint_path = os.path.join(train_path, game_name, 'models')
    estimator.restore(checkpoint_path)

    total_t = estimator.get_global_step()
    videoWriter = imageio.get_writer(
        'train_log/{}-{}.mp4'.format(game_name, total_t), fps=50)

    state = env.reset()
    r = 0
    tot = 0
    while True:
        image = env.render('rgb_array')
        videoWriter.append_data(image)
        q_value = estimator.predict(np.array([state]))
        action = np.argmax(q_value[0])
        state, reward, done, info = env.step(action)

        r += reward
        tot += 1
        if done:
            image = env.render('rgb_array')
            videoWriter.append_data(image)
            print(tot, r)
            break
    videoWriter.close()


if __name__ == '__main__':
    main()

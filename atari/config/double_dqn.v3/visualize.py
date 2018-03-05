import os
import cv2
import click
import numpy as np
import tensorflow as tf
from util import train_path
from wrapper import atari_env
from estimator import Estimator


@click.command()
@click.option('--game_name')
def visualize(game_name):
    videoWriter = cv2.VideoWriter('{}.mp4'.format(game_name),
                                  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                  60, (210, 160))
    env = atari_env(game_name, videowriter=videoWriter)

    checkpoint_path = os.path.join(train_path, game_name, 'models')
    estimator = Estimator(
        env.observation_space.shape,
        env.action_space.n,
        1e-4,
        update_target_rho=1.0)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    saver.restore(sess, latest_checkpoint)

    state = env.reset()
    while True:
        q_value = estimator.predict(sess, [state])
        action = np.argmax(q_value[0])
        state, reward, done, _ = env.step(action)
        if env.unwrapped.ale.lives() == 0:
            break


if __name__ == '__main__':
    visualize()

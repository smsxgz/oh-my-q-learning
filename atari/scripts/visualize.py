import os
import imageio
import click
import numpy as np

used_gpu = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu
import tensorflow as tf

from util import train_path
from wrapper import atari_env
from estimator import Estimator


@click.command()
@click.option('--game_name')
@click.option('--write_video', action="store_true")
def main(game_name, write_video):
    env = atari_env(game_name)

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

    def visualize():
        total_t = sess.run(tf.train.get_global_step())
        videoWriter = imageio.get_writer(
            '{}-{}.mp4'.format(game_name, total_t), fps=30)

        state = env.reset(videowriter=videoWriter)
        lives = env.unwrapped.ale.lives()
        r = 0
        while True:
            q_value = estimator.predict(sess, [state])
            action = np.argmax(q_value[0])
            state, reward, done, _ = env.step(action)
            r += reward
            if done:
                assert env.unwrapped.ale.lives() < lives
                lives = env.unwrapped.ale.lives()
                if lives == 0:
                    print(r)
                    break
                else:
                    state = env.reset()
        videoWriter.close()

    def evaluate():
        res = []
        for i in range(50):
            state = env.reset()
            lives = env.unwrapped.ale.lives()
            r = 0
            while True:
                q_value = estimator.predict(sess, [state])
                action = np.argmax(q_value[0])
                state, reward, done, _ = env.step(action)
                r += reward
                if done:
                    assert env.unwrapped.ale.lives() < lives
                    lives = env.unwrapped.ale.lives()
                    if lives == 0:
                        res.append(r)
                        break
                    else:
                        state = env.reset()
        print(sum(res) / 50)

    if write_video:
        visualize()
    else:
        evaluate()


if __name__ == '__main__':
    main()

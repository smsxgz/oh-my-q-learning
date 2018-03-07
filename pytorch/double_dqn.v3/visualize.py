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


def main(env, agent, write_video):
    def visualize():
        total_t = sess.run(tf.train.get_global_step())
        videoWriter = imageio.get_writer(
            '{}-{}.mp4'.format(game_name, total_t), fps=30)

        state = env.reset(videowriter=videoWriter)
        r = 0
        while not done:
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

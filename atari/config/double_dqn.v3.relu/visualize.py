import os
import click
import imageio
import numpy as np
from tqdm import tqdm

used_gpu = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu
import tensorflow as tf

from util import train_path
from wrapper import atari_env
from estimator import Estimator


@click.command()
@click.option('--game_name')
@click.option('--write_video', is_flag=True)
def main(game_name, write_video):
    env = atari_env(game_name)

    tf.Variable(0, name='global_step', trainable=False)
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
    checkpoint_path = os.path.join(train_path, game_name, 'models')
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    saver.restore(sess, latest_checkpoint)

    def visualize():
        total_t = sess.run(tf.train.get_global_step())
        videoWriter = imageio.get_writer(
            'train_log/{}-{}.mp4'.format(game_name, total_t), fps=30)

        state = env.reset(videowriter=videoWriter)
        lives = env.unwrapped.ale.lives()
        print(lives)
        r = 0
        tot = 0
        while True:
            q_value = estimator.predict(sess, [state])
            action = np.argmax(q_value[0])
            state, reward, done, _ = env.step(action)
            r += reward
            tot += 1
            if done:
                ale_lives = env.unwrapped.ale.lives()
                print(ale_lives)
                if ale_lives == 0 or lives == ale_lives:
                    print(tot, r)
                    break
                else:
                    lives = ale_lives
                    state = env.reset()
        videoWriter.close()

    def evaluate():
        res = []
        for i in tqdm(range(50)):
            state = env.reset()
            lives = env.unwrapped.ale.lives()
            r = 0
            while True:
                q_value = estimator.predict(sess, [state])
                action = np.argmax(q_value[0])
                state, reward, done, _ = env.step(action)
                r += reward
                if done:
                    ale_lives = env.unwrapped.ale.lives()
                    if ale_lives == 0 or lives == ale_lives:
                        res.append(r)
                        break
                    else:
                        lives = ale_lives
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

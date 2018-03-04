import cv2
import click
import tensorflow as tf
from estimator import Estimator
from wrapper import atari_env


@click.command()
@click.option('--game_name')
def visualize(game_name, checkpoint_path):
    videoWriter = cv2.VideoWriter('{}.mp4'.format(game_name),
                                  cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), 30,
                                  (210, 160))
    env = atari_env(game_name, videowriter=videoWriter)
    estimator = Estimator(
        env.state_shape, env.action_n, 1e-4, update_target_rho=1.0)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    saver.restore(sess, latest_checkpoint)

    while True:
        pass

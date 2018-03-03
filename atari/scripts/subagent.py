import zmq
import click
import msgpack
import numpy as np
import msgpack_numpy
from wrapper import atari_env
msgpack_numpy.patch()


class SubAgent(object):
    def __init__(self, game_name, identity, url):
        self.env = atari_env(game_name)
        self.identity = 'SubAgent-{}'.format(identity)
        self.url = url

        self.action_n = self.env.action_space.n
        self.allowed_actions = list(range(self.action_n))

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.identity = self.identity.encode('utf-8')
        socket.connect(self.url)

        # Reset env
        print('subagent {} start!'.format(self.identity))
        socket.send(
            msgpack.dumps((b'ready', self.action_n,
                           self.env.observation_space.shape)))

        while True:
            action = socket.recv()
            if action == b'reset':
                state = self.env.reset()
                game_reward = 0
                game_length = 0
                game_real_reward = 0
                socket.send(msgpack.dumps(state))
                continue

            if action == b'close':
                socket.close()
                context.term()
                break

            action = msgpack.loads(action)
            assert action in self.allowed_actions
            next_state, reward, done, _ = self.env.step(action)
            game_reward += reward
            game_length += 1
            game_real_reward += reward
            info = {}
            if done:
                next_state = self.env.reset()
                info = {'reward': game_reward, 'length': game_length}
                game_reward = 0
                game_length = 0
                if self.env.was_real_done:
                    info['real_reward'] = game_real_reward
                    game_real_reward = 0

            socket.send(
                msgpack.dumps((next_state, np.sign(reward), done, info)))


@click.command()
@click.option('--game_name')
@click.option('--identity')
def main(game_name, identity):
    s = SubAgent(game_name, identity,
                 'ipc://./.ipc/{}/Agent.ipc'.format(game_name))
    s.run()


if __name__ == '__main__':
    main()

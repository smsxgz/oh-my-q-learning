import zmq
import msgpack
import numpy as np
import msgpack_numpy
from lib.wrappers import make_env
from lib.util import str_reward
from multiprocessing import Process

msgpack_numpy.patch()


class AgentEnv(Process):
    def __init__(self, game_name, url, i):
        super(AgentEnv, self).__init__()
        self.env = make_env(game_name)
        self.rewards_stats = []
        self.url = url
        self.identity = ('Agent-%d' % i).encode('utf-8')

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.identity = self.identity
        socket.connect(self.url)

        state = self.env.reset()
        socket.send(msgpack.dumps(('reset', state)))
        episode_reward = 0
        while True:
            reply = socket.recv()
            if reply == b'reset':
                self.rewards_stats.append(episode_reward)
                print(str_reward(self.rewards_stats, 50))
                episode_reward = 0
                state = self.env.reset()
                socket.send(msgpack.dumps(('reset', state)))
            else:
                action = msgpack.loads(reply)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                socket.send(
                    msgpack.dumps(('step', state, action, np.sign(reward),
                                   next_state, done)))
                state = next_state

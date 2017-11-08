import zmq
import msgpack
import numpy as np
import msgpack_numpy
from threading import Thread
msgpack_numpy.patch()


class Agent(Thread):
    def __init__(self, make_env, i, url, memory_url):
        super(Agent, self).__init__(daemon=True)
        self.env = make_env()
        self.url = url
        self.memory_url = memory_url
        self.identity = ('Agent-%d' % i).encode('utf-8')

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.identity = self.identity
        socket.connect(self.url)

        memory = context.socket(zmq.PUSH)
        memory.connect(self.memory_url)

        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        while True:
            socket.send(msgpack.dumps(state))
            action = msgpack.loads(socket.recv())
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            episode_length += 1
            memory.send(
                msgpack.dumps(('Transition', state, action, np.sign(reward),
                               next_state, done)))
            state = next_state
            if done:
                memory.send(
                    msgpack.dumps(('Reward', episode_reward, episode_length)))
                episode_reward = 0
                episode_length = 0
                state = self.env.reset()

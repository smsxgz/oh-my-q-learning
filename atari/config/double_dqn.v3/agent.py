import os
import zmq
import msgpack
import numpy as np
import msgpack_numpy
from collections import OrderedDict

msgpack_numpy.patch()


class Agent(object):
    """Control agents' threads."""

    def __init__(self, num_agents, game_name):
        path = './.ipc/{}'.format(game_name)
        if os.path.exists(path):
            os.system('rm {} -rf'.format(path))
        os.makedirs(path)

        self.num_agents = num_agents
        for i in range(num_agents):
            os.system(
                'python subagent.py --game_name {} --identity {} &'.format(
                    game_name, i))

        url = 'ipc://./.ipc/{}/Agent.ipc'.format(game_name)
        self.context = zmq.Context()
        self.agent_socket = self.context.socket(zmq.ROUTER)
        self.agent_socket.bind(url)

        self.addrs = OrderedDict()
        self.action_n, self.state_shape = self._prepare()

    def _prepare(self):
        for _ in range(self.num_agents):
            addr, empty, msg = self.agent_socket.recv_multipart()
            msg = msgpack.loads(msg)
            self.addrs[addr] = None
            # print(msg)
            assert msg[0] == b'ready'
            action_n = msg[1]
            state_shape = msg[2]
        return action_n, state_shape

    def reset(self):
        for addr in self.addrs:
            self.agent_socket.send_multipart([addr, b'', b'reset'])
        for _ in range(self.num_agents):
            addr, empty, msg = self.agent_socket.recv_multipart()
            msg = msgpack.loads(msg)
            self.addrs[addr] = msg
        return np.array(list(self.addrs.values()))

    def step(self, actions):
        for action, addr in zip(actions, self.addrs.keys()):
            self.agent_socket.send_multipart(
                [addr, b'', msgpack.dumps(action)])

        info = {}
        for _ in range(self.num_agents):
            addr, empty, msg = self.agent_socket.recv_multipart()
            msg = msgpack.loads(msg)
            if msg[-1]:
                info[addr] = msg[-1]
            self.addrs[addr] = msg[:-1]
        states, rewards, dones = map(np.array, zip(*self.addrs.values()))
        return states, rewards, dones, info

    def close(self):
        for addr in self.addrs:
            self.agent_socket.send_multipart([addr, b'', b'close'])
        self.agent_socket.close()
        self.context.term()

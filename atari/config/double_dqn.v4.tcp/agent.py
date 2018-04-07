import os
import zmq
import msgpack
import numpy as np
import msgpack_numpy
from collections import OrderedDict

msgpack_numpy.patch()


class Agent(object):
    """Control agents' threads."""

    def __init__(self, num_agents, game_name, port=7878):
        self.game_name = game_name

        self.num_agents = num_agents
        command = 'python3 subagent.py --game_name {} --port {}'.format(
            game_name, port)
        for i in range(num_agents):
            os.system(command + ' --identity {} &'.format(i))

        url = "tcp://0.0.0.0:{}".format(port)
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

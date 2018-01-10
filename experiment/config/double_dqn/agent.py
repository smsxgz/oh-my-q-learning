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
        try:
            os.makedirs('./tmp/{}'.format(game_name))
        except Exception:
            pass

        self.num_agents = num_agents
        for i in range(num_agents):
            os.system('python subagent.py --game_name {} --identity {} &'.
                      format(game_name, i))

        url = 'ipc://./tmp/{}/Agent.ipc'.format(game_name)
        self.context = zmq.Context()
        self.agent_socket = self.context.socket(zmq.ROUTER)
        self.agent_socket.bind(url)

        self.addrs = OrderedDict()

    def agent_recv(self):
        for _ in range(self.num_agents):
            addr, empty, msg = self.agent_socket.recv_multipart()
            self.addrs[addr] = msgpack.loads(msg)
        return map(np.array, zip(*self.addrs.values()))

    def agent_send(self, actions):
        for action, addr in zip(actions, self.addrs.keys()):
            self.agent_socket.send_multipart(
                [addr, b'', msgpack.dumps(action)])

    def reset(self):
        for _ in range(self.num_agents):
            addr, empty, msg = self.agent_socket.recv_multipart()
            msg = msgpack.loads(msg)
            self.addrs[addr] = msg[0]
            action_n = msg[1]
        return np.array(list(self.addrs.values())), action_n

    def step(self, actions):
        self.agent_send(actions)
        return self.agent_recv()

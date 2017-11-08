import zmq
import msgpack
import msgpack_numpy
from multiprocessing import Process
from agent import Agent
msgpack_numpy.patch()


class SuperAgent(Process):
    """Control agents' threads."""

    def __init__(self, num_agents, make_env, master_url, memory_url, i):
        super(SuperAgent, self).__init__()
        self.num_agents = num_agents
        self.id = i
        self.identity = ('SuperAgent-%d' % i).encode('utf-8')
        self.master_url = master_url
        self.agent_url = 'ipc://./tmp/{}-agent-{}'.format(
            self.env.game_name, i)

        # Start (sub)agents
        agents = [
            Agent(make_env, j, self.agent_url, memory_url)
            for j in range(num_agents)
        ]
        [a.start() for a in agents]

    def agent_recv(self):
        addrs = []
        states = []
        for _ in range(self.num_agents):
            addr, empty, msg = self.agent_socket.recv()
            addrs.append(addr)
            states.append(msgpack.loads(msg))
        return states, addrs

    def agent_send(self, actions, addrs):
        for action, addr in zip(actions, addrs):
            self.agent_socket.send_multipart(
                [addr, b'', msgpack.dumps(action)])

    def run(self):
        self.context = zmq.Context()
        self.agent_socket = self.context.socket(zmq.ROUTER)
        self.agent_socket.bind(self.agent_url)

        master_socket = self.context.socket(zmq.REQ)
        master_socket.connet(self.master_url)

        while True:
            states, addrs = self.agent_recv()
            master_socket.send(msgpack.dumps(states))

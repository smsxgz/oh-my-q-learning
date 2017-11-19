import zmq
import msgpack
import msgpack_numpy
from threading import Thread
from multiprocessing import Process
msgpack_numpy.patch()


class Agent(Thread):
    def __init__(self, make_env, identity, url, memory_url):
        super(Agent, self).__init__(daemon=True)
        self.env = make_env()
        self.identity = identity
        self.url = url
        self.memory_url = memory_url

        self.allowed_actions = list(range(self.env.action_n))

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.identity = self.identity.encode('utf-8')
        socket.connect(self.url)

        memory = context.socket(zmq.PUSH)
        memory.connect(self.memory_url)

        state = self.env.reset()
        while True:
            socket.send(msgpack.dumps(state))
            action = msgpack.loads(socket.recv())

            # Do nothing, just assert noisily
            assert action in self.allowed_actions

            next_state, reward, done, _ = self.env.step()
            memory.send(
                msgpack.dumps((self.identity, state, action, reward,
                               next_state, done)))
            if done:
                state = self.env.reset()
            else:
                state = next_state


class SuperAgent(Process):
    """Control agents' threads."""

    def __init__(self, num_agents, make_env, master_url, memory_url, i):
        super(SuperAgent, self).__init__(daemon=True)
        self.num_agents = num_agents
        self.identity = 'SuperAgent-{:d}'.format(i)
        self.master_url = master_url
        self.url = master_url + '-superagent-{:d}'.format(i)

        # Start agents
        agents = [
            Agent(make_env, self.identity + '-Agent-{:d}'.format(j), self.url,
                  memory_url) for j in range(num_agents)
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
        master_socket.identity = self.identity.encode('utf-8')
        master_socket.connet(self.master_url)

        while True:
            states, addrs = self.agent_recv()
            master_socket.send(msgpack.dumps(states))
            actions = msgpack.dumps(master_socket.recv())
            self.agent_send(actions, addrs)

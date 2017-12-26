import zmq
import msgpack
import numpy as np
import msgpack_numpy
from threading import Thread
msgpack_numpy.patch()


class SubAgent(Thread):
    def __init__(self, make_env, identity, port, memory_url):
        super(SubAgent, self).__init__(daemon=True)
        self.env = make_env()
        self.identity = identity
        self.url = 'tcp://127.0.0.1:{}'.format(port)
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

            next_state, reward, done, _ = self.env.step(action)
            memory.send(
                msgpack.dumps((self.identity, state, action, reward,
                               next_state, done)))
            if done:
                state = self.env.reset()
                print(self.identity + ' done!')
            else:
                state = next_state


class Agent(object):
    """Control agents' threads."""

    def __init__(self, num_agents, make_env, master_url, memory_url, i):
        # super(SuperAgent, self).__init__(daemon=True)
        self.num_agents = num_agents
        self.identity = 'SuperAgent-{:d}'.format(i)
        self.master_url = master_url
        port = np.random.randint(1, 10000)
        self.url = 'tcp://*:{}'.format(port)

        # Start agents
        agents = [
            SubAgent(make_env, self.identity + '-Agent-{:d}'.format(j), port,
                     memory_url) for j in range(num_agents)
        ]
        [a.start() for a in agents]

    def agent_recv(self):
        addrs = []
        states = []
        for _ in range(self.num_agents):
            addr, empty, msg = self.agent_socket.recv_multipart()
            addrs.append(addr)
            state = msgpack.loads(msg)
            assert state.shape == (84, 84, 4)
            states.append(state)
        try:
            states = np.array(states)
        except Exception:
            from IPython import embed
            embed()
        return states, addrs

    def agent_send(self, actions, addrs):
        for action, addr in zip(actions, addrs):
            self.agent_socket.send_multipart(
                [addr, b'', msgpack.dumps(action)])

    def run(self):
        self.context = zmq.Context()
        self.agent_socket = self.context.socket(zmq.ROUTER)
        self.agent_socket.bind(self.url)

        master_socket = self.context.socket(zmq.REQ)
        master_socket.identity = self.identity.encode('utf-8')
        master_socket.connect(self.master_url)

        while True:
            states, addrs = self.agent_recv()
            master_socket.send(msgpack.dumps(states))
            actions = msgpack.loads(master_socket.recv())
            self.agent_send(actions, addrs)


def main():
    from lib.ale_wrapper import wrap_env
    import numpy as np

    master_url = 'ipc://./tmp/Master.ipc'
    memory_url = 'ipc://./tmp/Memory.ipc'

    def make_env():
        env = wrap_env('beam_rider')
        return env

    c = Agent(8, make_env, master_url, memory_url, np.random.randint(0, 100))
    c.run()


if __name__ == '__main__':
    main()

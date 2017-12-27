import zmq
import msgpack
import numpy as np
import msgpack_numpy
from threading import Thread
from env import make_env
msgpack_numpy.patch()


class SubAgent(Thread):
    def __init__(self, make_env, identity, url):
        super(SubAgent, self).__init__(daemon=True)
        self.env = make_env()
        self.identity = identity
        self.url = url

        self.allowed_actions = list(range(self.env.action_n))

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.identity = self.identity.encode('utf-8')
        socket.connect(self.url)

        # Reset env
        game_reward = 0
        state = self.env.reset()
        socket.send(msgpack.dumps(state))

        while True:
            action = msgpack.loads(socket.recv())

            # Do nothing, just assert noisily
            assert action in self.allowed_actions

            next_state, reward, done, _ = self.env.step(action)
            game_reward += reward
            info = None
            if done:
                next_state = self.env.reset()
                info = game_reward
                game_reward = 0

            socket.send(msgpack.dumps((next_state, reward, info)))


class Agent(object):
    """Control agents' threads."""

    def __init__(self, num_agents, make_env, game_name, master_url, i):
        self.num_agents = num_agents
        self.identity = 'SuperAgent-{:d}'.format(i)
        self.master_url = master_url
        self.url = 'ipc://./tmp/game_name/SuperAgent-{:d}.ipc'.format(i)

        def get_env():
            env = make_env(game_name)
            return env

        # Start agents
        agents = [
            SubAgent(get_env, '{}-Agent-{:d}'.format(self.identity, j),
                     self.url) for j in range(num_agents)
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

    import numpy as np

    master_url = 'ipc://./tmp/Master.ipc'

    c = Agent(8, make_env, master_url, np.random.randint(0, 100))
    c.run()


if __name__ == '__main__':
    main()

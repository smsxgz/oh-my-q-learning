import zmq
import msgpack
import numpy as np
import msgpack_numpy
from threading import Thread
from framework.master import Master
from lib.ale_wrapper import wrap_env
from zmq.eventloop.ioloop import IOLoop
from framework.agent import SuperAgent as Agent

msgpack_numpy.patch()


def random_worker(url, i, action_n):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    identity = 'Worker-{:d}'.format(i)
    socket.identity = identity.encode('utf-8')
    socket.connet(url)

    socket.send(b'READY')
    while True:
        address, empty, request = socket.recv_multipart()
        actions = np.random.randint(
            0, action_n, size=msgpack.loads(request).shape[0])
        socket.send_multipart([address, b'', msgpack.dumps(actions)])


def make_env():
    env = wrap_env('beam_rider')
    return env


def memory(memory_url):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect(memory_url)

    while True:
        socket.recv()


if __name__ == '__main__':
    master_url = 'ipc://./tmp/Master.ipc'
    worker_url = 'ipc://./tmp/Worker.ipc'
    memory_url = 'ipc://./tmp/Memory.ipc'
    for i in range(4):
        w = Thread(target=random_worker, args=(worker_url, i, 9))
        w.daemon = True
        w.start()

    for i in range(4):
        c = Agent(8, make_env, master_url, memory_url, i)
        c.start()

    Thread(target=memory, args=(memory_url, )).start()

    Master(worker_url, master_url)
    IOLoop.instance().start()

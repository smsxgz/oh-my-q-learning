import zmq
import time
import msgpack
import numpy as np
import msgpack_numpy
from threading import Thread
from framework.master import Master
from lib.ale_wrapper import wrap_env
from zmq.eventloop.ioloop import IOLoop

msgpack_numpy.patch()


def random_worker(url, i, action_n):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    identity = 'Worker-{:d}'.format(i)
    socket.identity = identity.encode('utf-8')
    socket.connect(url)

    socket.send(b'READY')
    tot = 0
    while True:
        address, empty, request = socket.recv_multipart()
        tot += 1
        if tot % 100 == 0:
            print(identity + ' get {:d} messages.'.format(tot))
        actions = np.random.randint(
            0, action_n, size=msgpack.loads(request).shape[0])
        time.sleep(0.01)
        socket.send_multipart([address, b'', msgpack.dumps(actions)])


def make_env():
    env = wrap_env('beam_rider')
    return env


def memory(memory_url):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(memory_url)

    tot = 0
    while True:
        socket.recv_multipart()
        tot += 1
        if tot % 100 == 0:
            print('Get {:d} memories.'.format(tot))


if __name__ == '__main__':
    master_url = 'ipc://./tmp/Master.ipc'
    worker_url = 'ipc://./tmp/Worker.ipc'
    memory_url = 'ipc://./tmp/Memory.ipc'
    for i in range(8):
        w = Thread(target=random_worker, args=(worker_url, i, 9))
        w.daemon = True
        w.start()

    Thread(target=memory, args=(memory_url, )).start()

    Master(worker_url, master_url)
    print('Initial finished.')
    IOLoop.instance().start()

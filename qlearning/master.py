import zmq
import random
import msgpack
import numpy as np
import msgpack_numpy
from collections import deque
from lib.util import Transition
from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream
msgpack_numpy.patch()


class Master(object):
    """A broker for DQN, but I would like to call it master!!!"""

    def __init__(self, backend_socket, frontend_socket, batch_size,
                 estimator_update_callable):
        self.available_workers = 0
        self.workers = []

        self.batch_size = batch_size
        self.estimator_update = estimator_update_callable

        self.backend = ZMQStream(backend_socket)
        self.frontend = ZMQStream(frontend_socket)
        self.backend.on_recv(self.handle_backend)

        self.loop = IOLoop.instance()

    def handle_backend(self, msg):
        # Queue worker address for LRU routing
        worker_addr, empty, client_addr = msg[:3]

        # add worker back to the list of workers
        self.available_workers += 1
        self.workers.append(worker_addr)

        # Third frame is READY or else a client reply address
        # If client reply, send rest back to frontend
        if client_addr != b"READY":
            empty, reply = msg[3:]
            self.frontend.send_multipart([client_addr, b'', reply])

        if self.available_workers == 1:
            # on first recv, start accepting frontend messages
            self.frontend.on_recv(self.handle_frontend)

    def handle_frontend(self, msg):
        # Now get next client request, route to LRU worker
        # Client request is [address][empty][request]
        client_addr, empty, request = msg
        request = msgpack.loads(request)
        if request[0] == 'reset':
            state = request[1]
            msg = [b'', client_addr, b'', msgpack.dumps(state)]
            self.worker_send(msg)
        elif request[0] == 'step':
            t = Transition(*request[1:])
            self.update(t)

            if t.done:
                self.frontend.send_multipart([client_addr, b'', b'reset'])
            else:
                msg = [b'', client_addr, b'', msgpack.dumps(t.next_state)]
                self.worker_send(msg)

    def worker_send(self, msg):
        #  Dequeue and drop the next worker address
        self.available_workers -= 1
        worker_id = self.workers.pop(0)

        self.backend.send_multipart([worker_id] + msg)
        if self.available_workers == 0:
            # stop receiving until workers become available again
            self.frontend.stop_on_recv()


class OnMaster(Master):
    """For online learning."""

    def __init__(self, **kwargs):
        super(OffMaster, self).__init__(**kwargs)
        self.memory = []

    def update(self, transition):
        self.memory.append(transition)
        if len(self.memory) == self.batch_size:
            samples = map(np.array, zip(*self.memory))
            self.estimator_update(*samples)
            self.memory = []


class OffMaster(Master):
    """For memory buffer learning."""

    def __init__(self, init_memory_size, memory_size, estimator_update_every,
                 **kwargs):
        super(OffMaster, self).__init__(**kwargs)
        self.init_memory_size = init_memory_size
        self.memory = deque(maxlen=memory_size)
        self.estimator_update_every = estimator_update_every
        self.tot = 0

    def update(self, transition):
        self.memory.append(transition)
        self.tot += 1
        if len(self.memory) > self.init_memory_size and \
                self.tot % self.estimator_update_every == 0:
            samples = random.sample(self.memory, self.batch_size)
            samples = map(np.array, zip(*samples))
            self.estimator_update(*samples)


def estimator_worker(url, i, sess, q_estimator, policy):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    identity = ('Worker-%d' % i).encode('utf-8')
    socket.identity = identity
    socket.connect(url)

    socket.send(b'READY')
    while True:
        address, empty, request = socket.recv_multipart()
        q_values = q_estimator.predict(sess, [msgpack.loads(request)])
        action = policy(q_values)
        socket.send_multipart([address, b'', msgpack.dumps(action)])

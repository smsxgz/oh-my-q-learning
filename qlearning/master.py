import msgpack
import numpy as np
import msgpack_numpy
from lib.util import Transition
from zmq.eventloop.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream
msgpack_numpy.patch()


class DQNMaster(object):
    """A broker for DQN, but I would like to call it master!!!"""

    def __init__(self, backend_socket, frontend_socket, batch_size,
                 callable_update):
        self.available_workers = 0
        self.workers = []

        self.memory = []
        self.batch_size = batch_size
        self.update = callable_update

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
            self.memory.append(t)
            if len(self.memory) == self.batch_size:
                samples = map(np.array, zip(*self.memory))
                self.update(*samples)
                self.memory = []

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

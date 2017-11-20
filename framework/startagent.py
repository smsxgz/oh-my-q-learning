from framework.agent import SuperAgent as Agent
from lib.ale_wrapper import wrap_env
import numpy as np

master_url = 'ipc://./tmp/Master.ipc'
worker_url = 'ipc://./tmp/Worker.ipc'
memory_url = 'ipc://./tmp/Memory.ipc'

def make_env():
    env = wrap_env('beam_rider')
    return env

c = Agent(8, make_env, master_url, memory_url, np.random.randint(0,100))
c.run()

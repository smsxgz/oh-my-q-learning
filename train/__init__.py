import math
import tensorflow as tf
path = 'test'
used_gpu = '0'
num_agent = 8
num_worker = 4
batch_size = 256
discount_factor = 0.95
learning_rate = 1e-4

epsilon_start = 0.5
epsilon_end = 0.05
epsilon_decay_steps = 500000

update_target_estimator_every = 1000
save_model_every = 100

init_memory_size = 50000
memory_size = 500000

vmin = -10.0
vmax = 10.0
N = 51


# Network function for some simple games and mujoco
def network_struct(state_n, action_n):
    def network(x_shape):
        X_pl = tf.placeholder(shaoe=x_shape, dtype=tf.float32, name="X")
        fc1 = tf.contrib.layers.fully_connected(X_pl, 10 * state_n)
        fc2 = tf.contrib.layers.fully_connected(
            fc1, int(10 * math.sqrt(state_n * action_n)) + 1)
        return X_pl, tf.contrib.layers.fully_connected(fc2, 10 * action_n)

    return network

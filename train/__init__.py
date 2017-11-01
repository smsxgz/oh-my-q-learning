import math
import tensorflow as tf

path = 'test'
used_gpu = '0'
num_agent = 8
num_worker = 4
batch_size = 32
discount_factor = 0.99
learning_rate = 6.25e-5

epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay_steps = 500000

update_estimator_every = num_agent * 4
update_target_estimator_every = 1000
save_model_every = 100

init_memory_size = 80000
memory_size = 1000000

vmin = -10.0
vmax = 10.0
N = 51


# Network function for some simple games
def network_struct(state_n, action_n):
    def network(x_shape):
        X_pl = tf.placeholder(shape=x_shape, dtype=tf.float32, name="X")
        fc1 = tf.contrib.layers.fully_connected(X_pl, 10 * state_n)
        fc2 = tf.contrib.layers.fully_connected(
            fc1, int(10 * math.sqrt(state_n * action_n)) + 1)
        return X_pl, tf.contrib.layers.fully_connected(fc2, 10 * action_n)

    return network

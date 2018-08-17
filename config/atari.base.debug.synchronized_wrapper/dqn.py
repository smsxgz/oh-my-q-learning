import os
import time
import pickle
import numpy as np
from memory import Memory
from collections import defaultdict
from tensorboardX import SummaryWriter


class ResultsBuffer(object):
    def __init__(self, rewards_history=[]):
        self.buffer = defaultdict(list)
        assert isinstance(rewards_history, list)
        self.rewards_history = rewards_history

    def update_infos(self, info, total_t):
        for key in info:
            msg = info[key]
            self.buffer['reward'].append(msg[b'reward'])
            self.buffer['length'].append(msg[b'length'])
            if b'real_reward' in msg:
                self.buffer['real_reward'].append(msg[b'real_reward'])
                self.buffer['real_length'].append(msg[b'real_length'])
                self.rewards_history.append(
                    [total_t, key, msg[b'real_reward']])

    def update_summaries(self, summaries):
        for key in summaries:
            self.buffer[key].append(summaries[key])

    def add_summary(self, summary_writer, total_t, time):
        s = {'time': time}
        for key in self.buffer:
            if self.buffer[key]:
                s[key] = np.mean(self.buffer[key])
                self.buffer[key].clear()

        for key in s:
            summary_writer.add_scalar(key, s[key], total_t)


def dqn(env,
        model,
        base_path,
        batch_size=32,
        epsilon=0.01,
        save_model_every=1000,
        update_target_every=1000,
        learning_starts=200,
        memory_size=500000,
        num_iterations=6250000):
    events_path = os.path.join(base_path, 'events')
    models_path = os.path.join(base_path, 'models')
    if not os.path.exists(events_path):
        os.makedirs(events_path)
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    model.load_model(models_path)
    summary_writer = SummaryWriter(events_path)
    rewards_history = []
    pkl_path = '{}/rewards.pkl'.format(base_path)
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            rewards_history = pickle.load(f)

    memory_buffer = Memory(memory_size)
    results_buffer = ResultsBuffer(rewards_history)
    global_step = model.get_global_step()

    try:
        states = env.reset()
        for i in range(learning_starts):
            actions = model.get_action(states, epsilon)
            next_states, rewards, dones, info = env.step(actions)

            memory_buffer.extend(
                zip(states, actions, rewards, next_states, dones))
            states = next_states

        states = env.reset()
        start = time.time()
        for i in range(num_iterations):
            actions = model.get_action(states, epsilon)
            next_states, rewards, dones, info = env.step(actions)

            results_buffer.update_infos(info, global_step)
            memory_buffer.extend(
                zip(states, actions, rewards, next_states, dones))

            global_step, summaries = model.update(
                *memory_buffer.sample(batch_size))
            results_buffer.update_summaries(summaries)

            if global_step % update_target_every == 0:
                model.update_target()

            if global_step % save_model_every == 0:
                t = time.time() - start
                model.save_model(models_path)
                if global_step > 0:
                    print("Save model, global_step: {}, delta_time: {}.".format(
                        global_step, t))
                results_buffer.add_summary(summary_writer, global_step, t)
                start = time.time()

            states = next_states

    except Exception as e:
        raise e

    finally:
        model.save_model(models_path)
        with open(pkl_path, 'wb') as f:
            pickle.dump(results_buffer.rewards_history, f)

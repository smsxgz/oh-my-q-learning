import os
import time
import numpy as np
import pickle
import json
from util import Memory
from collections import defaultdict

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
                self.rewards_history.append([total_t, key, msg[b'real_reward']])

    def update_summaries(self, summaries, total_t):
        #loss, max_q_value, min_q_value = summaries
        loss = summaries
        self.buffer['loss'].append(loss)
        #self.buffer['max_q_value'].append(max_q_value)
        #self.buffer['min_q_value'].append(min_q_value)

    def add_summary(self, summary_writer, total_t, time):
        s = {'time': time}
        for key in self.buffer:
            if self.buffer[key]:
                s[key] = np.mean(self.buffer[key])
                self.buffer[key].clear()

        for key in s:
            summary_writer.add_scalar(key, s[key], total_t)

def distdqn(env, model, batch_size, summary_writer, checkpoint_path, save_model_every=1000, 
        update_target_every=1000, learning_starts=200, memory_size=500000, num_iterations=6250000):
    if os.path.exists('{}/global_step.json'.format(checkpoint_path)):
        global_step = json.load(open('{}/global_step.json'.format(checkpoint_path))) + 1
        model.load_model(checkpoint_path)
    else:
        global_step = 0

    rewards_history = []
    pkl_path = 'train_log/{}/rewards.pkl'.format(env.game_name)
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            rewards_history = pickle.load(f)

    memory_buffer = Memory(memory_size)
    results_buffer = ResultsBuffer(rewards_history)

    try:
        states = env.reset()
        for i in range(learning_starts):
            actions = model.get_action(states)
            next_states, rewards, dones, info = env.step(actions)
            
            memory_buffer.extend(zip(states, actions, rewards, next_states, dones))
            states = next_states

        states = env.reset()
        start = time.time()
        for i in range(num_iterations):
            actions = model.get_action(states)
            next_states, rewards, dones, info = env.step(actions)
            
            results_buffer.update_infos(info, global_step)
            memory_buffer.extend(zip(states, actions, rewards, next_states, dones))

            summaries = model.update(*memory_buffer.sample(batch_size))
            #print('Global step: {}, Loss: {}'.format(global_step, summaries))
            global_step += 1
            json.dump(global_step, open('{}/global_step.json'.format(checkpoint_path), 'w'))
            results_buffer.update_summaries(summaries, global_step)

            if global_step % update_target_every == 0:
                model.update_target()

            if global_step % save_model_every == 0:
                t = time.time() - start
                model.save_model(checkpoint_path, global_step)
                print("Save model, global_step: {}, delta_time: {}.".format(global_step, t))
                results_buffer.add_summary(summary_writer, global_step, t)
                start = time.time()

            states = next_states
    except Exception as e:
        raise e

    finally:
        model.save_model(checkpoint_path, global_step)
        with open(pkl_path, 'wb') as f:
            pickle.dump(results_buffer.rewards_history, f)


import os
import time
import pickle
import numpy as np
import tensorflow as tf
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
                self.rewards_history.append(
                    [total_t, key, msg[b'real_reward']])

    def update_summaries(self, summaries, total_t):
        loss, max_q_value, min_q_value = summaries
        self.buffer['loss'].append(loss)
        self.buffer['max_q_value'].append(max_q_value)
        self.buffer['min_q_value'].append(min_q_value)

    def add_summary(self, summary_writer, total_t, time):
        summary_writer.add_scalars(
            '', {
                'time': time,
                'loss': np.mean(self.buffer['loss']),
                'max_q_value': np.mean(self.buffer['max_q_value']),
                'min_q_value': np.mean(self.buffer['min_q_value'])
            }, total_t)
        if self.buffer['reward']:
            summary_writer.add_scalars(
                'game', {
                    'reward': np.mean(self.buffer['reward']),
                    'length': np.mean(self.buffer['length'])
                }, total_t)
            self.buffer['reward'].clear()
            self.buffer['length'].clear()

        if self.buffer['real_reward']:
            summary_writer.add_scalars(
                'game', {
                    'real_reward': np.mean(self.buffer['real_reward']),
                    'real_length': np.mean(self.buffer['real_length'])
                }, total_t)
            self.buffer['real_reward'].clear()
            self.buffer['real_length'].clear()


def dqn(sess,
        env,
        estimator,
        batch_size,
        summary_writer,
        checkpoint_path,
        exploration_policy_fn,
        discount_factor=0.99,
        save_model_every=1000,
        update_target_every=1,
        learning_starts=100,
        memory_size=100000,
        num_iterations=500000):

    estimator.restore(sess, checkpoint_path)

    rewards_history = []
    pkl_path = 'train_log/{}/rewards.pkl'.format(env.game_name)
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            rewards_history = pickle.load(f)

    total_t = sess.run(tf.train.get_global_step())

    memory_buffer = Memory(memory_size)
    results_buffer = ResultsBuffer(rewards_history)

    try:
        states = env.reset()
        for i in range(learning_starts):
            q_values = estimator.predict(sess, states)
            actions = exploration_policy_fn(q_values, total_t)
            next_states, rewards, dones, _ = env.step(actions)

            memory_buffer.extend(
                zip(states, actions, rewards, next_states, dones))
            states = next_states

        states = env.reset()
        start = time.time()
        for i in range(num_iterations):
            q_values = estimator.predict(sess, states)
            actions = exploration_policy_fn(q_values, total_t)
            next_states, rewards, dones, info = env.step(actions)

            results_buffer.update_infos(info, total_t)
            memory_buffer.extend(
                zip(states, actions, rewards, next_states, dones))

            states_batch, action_batch, reward_batch, next_states_batch, \
                done_batch = memory_buffer.sample(batch_size)

            # compute target batch (y)
            batch_size = states_batch.shape[0]
            best_actions = np.argmax(
                estimator.predict(sess, next_states_batch), axis=1)

            q_values_next_target = estimator.target_predict(
                sess, next_states_batch)
            discount_factor_batch = np.invert(done_batch).astype(
                np.float32) * discount_factor
            targets_batch = reward_batch + discount_factor_batch * \
                q_values_next_target[np.arange(batch_size), best_actions]

            # update
            _, total_t, *summaries = estimator.update(
                sess, states_batch, action_batch, targets_batch)
            results_buffer.update_summaries(summaries, total_t)

            if total_t % update_target_every == 0:
                estimator.target_update(sess)

            if total_t % save_model_every == 0:
                t = time.time() - start
                estimator.save(
                    sess,
                    os.path.join(checkpoint_path, 'model'),
                    total_t,
                    write_meta_graph=False)
                print("Save session, global_step: {}, delta_time: {}.".format(
                    total_t, t))

                results_buffer.add_summary(summary_writer, total_t, t)
                start = time.time()

            states = next_states

    except Exception as e:
        raise e

    finally:
        estimator.save(sess, os.path.join(checkpoint_path, 'model'), total_t)

        with open(pkl_path, 'wb') as f:
            pickle.dump(results_buffer.rewards_history, f)

import os
import time
import traceback
import numpy as np
from util import Memory
from collections import defaultdict


def dqn(env,
        estimator,
        batch_size,
        summary_writer,
        exploration_policy_fn,
        discount_factor=0.99,
        update_target_every=1,
        learning_starts=100,
        memory_size=100000,
        num_iterations=500000,
        update_every=4):

    total_t = 0
    score = 0
    episode = 0
    memory_buffer = Memory(memory_size)

    try:
        states = env.reset()
        for i in range(learning_starts):
            q_values = estimator.predict(states.reshape((1, ) + states.shape))
            actions = exploration_policy_fn(q_values, total_t)
            next_states, rewards, dones, _ = env.step(actions)
            score += rewards

            memory_buffer.append((states, actions, rewards, next_states, dones))
            states = next_states

            if dones:
                states = env.reset()
                episode += 1
                summary_writer.add_scalar('tot_reward', score, total_t)
                summary_writer.add_scalar('epi_reward', score, episode)
                score = 0

            total_t += 1

        states = env.reset()
        for i in range(num_iterations):
            q_values = estimator.predict(states.reshape((1, ) + states.shape))
            actions = exploration_policy_fn(q_values, total_t)
            next_states, rewards, dones, info = env.step(actions)
            score += rewards

            if total_t % update_every == 0:
                memory_buffer.append((states, actions, rewards, next_states, dones))
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = memory_buffer.sample(batch_size)

                # compute target batch
                batch_size = states_batch.shape[0]
                best_actions = np.argmax(estimator.predict(next_states_batch), axis=1)

                q_values_next_target = estimator.target_predict(next_states_batch)
                discount_factor_batch = np.invert(done_batch).astype(np.float32) * discount_factor
                targets_batch = reward_batch + discount_factor_batch * \
                    q_values_next_target[np.arange(batch_size), best_actions]

                # update
                info = estimator.update(states_batch, action_batch, targets_batch)
                for k in info:
                    summary_writer.add_scalar(k, info[k], total_t)

            if total_t % update_target_every == 0:
                estimator.target_update()

            states = next_states
            if dones:
                states = env.reset()
                episode += 1
                summary_writer.add_scalar('tot_reward', score, total_t)
                summary_writer.add_scalar('epi_reward', score, episode)
                score = 0
            total_t += 1

    except KeyboardInterrupt:
        print("\nKeyboard interrupt!")

    except Exception:
        traceback.print_exc()

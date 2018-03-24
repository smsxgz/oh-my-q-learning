import os
import time
import pickle
import numpy as np
from util import Memory, ResultsBuffer


def train(env,
          basename,
          model,
          batch_size,
          summary_writer,
          checkpoint_path,
          exploration_policy_fn,
          gamma=0.99,
          update_target_every=1,
          learning_starts=100,
          memory_size=100000,
          num_iterations=500000,
          update_summaries_every=1000,
          save_model_every=10000,
          update_leaf_every=1):

    model.restore(checkpoint_path)

    rewards_history = []
    pkl_path = 'train_log/{}/rewards.pkl'.format(basename)
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            rewards_history = pickle.load(f)

    total_t = model.get_global_step()

    memory_buffer = Memory(memory_size)
    results_buffer = ResultsBuffer(rewards_history)

    try:
        states = env.reset()
        start = time.time()
        for i in range(num_iterations):
            actions = model.get_action(states)
            actions = exploration_policy_fn(actions, total_t)
            next_states, rewards, dones, info = env.step(actions)

            results_buffer.update_infos(info, total_t)
            memory_buffer.extend(zip(states, actions, rewards, next_states, dones))
            states = next_states

            if i >= learning_starts:
                # update
                total_t, summaries = model.update(gamma, *memory_buffer.sample(batch_size))
                results_buffer.update_summaries(summaries, total_t)

                if total_t % update_leaf_every == 0:
                    model.update_leaf(gamma, *memory_buffer.sample(batch_size))

                if total_t % update_target_every == 0:
                    model.update_target()

                if total_t % update_summaries_every == 0:
                    t = time.time() - start
                    print("global_step: {}, delta_time: {}.".format(total_t, t))
                    results_buffer.add_summary(summary_writer, total_t, t)
                    start = time.time()

                if total_t % save_model_every == 0:
                    model.save(checkpoint_path)
                    print("save model...")

    except Exception as e:
        raise e
    finally:
        model.save(checkpoint_path)
        with open(pkl_path, 'wb') as f:
            pickle.dump(results_buffer.rewards_history, f)

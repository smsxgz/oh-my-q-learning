import numpy as np
from util import Memory


def dqn(env,
        estimator,
        batch_size,
        summary_writer,
        exploration_policy_fn,
        gamma=0.99,
        update_target_every=1,
        warm_up=100,
        memory_size=100000,
        num_iterations=500000,
        update_every=4):

    score = 0
    episode = 0
    memory = Memory(memory_size)

    state = env.reset()
    for total_t in range(num_iterations):
        q_values = estimator.predict(state.reshape((1, ) + state.shape))
        action = exploration_policy_fn(q_values, total_t)
        next_state, reward, done, info = env.step(action)
        score += reward
        memory.append((state, action, reward, next_state, done))

        if total_t % update_every == 0 and total_t > warm_up:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)

            # compute target batch
            batch_size = state_batch.shape[0]
            best_actions = np.argmax(estimator.predict(next_state_batch), axis=1)

            q_values_next_target = estimator.predict(next_state_batch, target=True)
            discount_factor_batch = np.invert(done_batch).astype(np.float32) * gamma
            target_batch = reward_batch + discount_factor_batch * \
                q_values_next_target[np.arange(batch_size), best_actions]

            # update
            info = estimator.update(state_batch, action_batch, target_batch)
            for k in info:
                summary_writer.add_scalar('update_info/' + k, info[k], total_t)

        if total_t % update_target_every == 0:
            estimator.target_update()

        state = next_state
        if done:
            state = env.reset()
            episode += 1
            summary_writer.add_scalar('episode_info/tot_reward', score, total_t)
            summary_writer.add_scalar('episode_info/epi_reward', score, episode)
            score = 0

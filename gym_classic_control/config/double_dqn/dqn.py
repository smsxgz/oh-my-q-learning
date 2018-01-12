import tensorflow as tf


def dqn(sess,
        env,
        q_estimator,
        target_estimator,
        memory_buffer,
        exploration_policy_fn,
        num_episodes=2000):

    total_t = sess.run(tf.contrib.framework.get_global_step())
    states = env.reset()
    while True:
        q_values = q_estimator.predict(sess, states)
        actions = exploration_policy_fn(q_values, total_t)
        next_states, rewards, dones, info = env.step(actions)

        memory_buffer.extend(zip(states, actions, rewards, next_states, dones))

        sample = memory_buffer.sample()
        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(
            simple_value=stats.episode_rewards[i_episode],
            node_name="episode_reward",
            tag="episode_reward")
        episode_summary.value.add(
            simple_value=stats.episode_lengths[i_episode],
            node_name="episode_length",
            tag="episode_length")
        q_estimator.summary_writer.add_summary(episode_summary, total_t)
        q_estimator.summary_writer.flush()

        yield total_t, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode + 1],
            episode_rewards=stats.episode_rewards[:i_episode + 1])

    env.close()
    return stats

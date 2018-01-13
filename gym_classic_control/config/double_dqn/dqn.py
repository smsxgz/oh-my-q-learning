import tensorflow as tf


def dqn(sess,
        env,
        update_fn,
        q_estimator,
        memory_buffer,
        summary_writer,
        exploration_policy_fn,
        num_iterations=2000000):

    total_t = sess.run(tf.train.get_global_step())
    states = env.reset()
    for i in range(num_iterations):
        try:
            q_values = q_estimator.predict(sess, states)
            actions = exploration_policy_fn(q_values, total_t)
            next_states, rewards, dones, info = env.step(actions)

            memory_buffer.extend(zip(states, actions, rewards, next_states, dones))
            sample = memory_buffer.sample()
            if sample:
                total_t = update_fn(*sample)

            states = next_states

            # Add summaries to tensorboard
            if info:
                mean_reward = sum(
                    list(item.values())[0] for item in info) / len(info)
                episode_summary = tf.Summary()
                episode_summary.value.add(
                    simple_value=mean_reward, node_name="rewards", tag="rewards")

                summary_writer.add_summary(episode_summary, total_t)
                summary_writer.flush()
        except KeyboardInterrupt:
            break
            
    env.close()

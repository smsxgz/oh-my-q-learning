import os
import traceback
import numpy as np
import tensorflow as tf


def dqn(sess,
        env,
        estimator,
        memory_buffer,
        summary_writer,
        checkpoint_path,
        exploration_policy_fn,
        discount_factor=0.99,
        save_model_every=1000,
        update_target_every=1,
        learning_starts=100,
        num_iterations=500000):

    saver = tf.train.Saver(max_to_keep=num_iterations // save_model_every)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    if latest_checkpoint:
        print("Loading model checkpoint {}...".format(latest_checkpoint))
        try:
            saver.restore(sess, latest_checkpoint)
        except Exception:
            print('Loading failed, we will Start from scratch!!')

    total_t = sess.run(tf.train.get_global_step())

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
        rewards_buffer = []
        for i in range(num_iterations):
            q_values = estimator.predict(sess, states)
            actions = exploration_policy_fn(q_values, total_t)
            next_states, rewards, dones, info = env.step(actions)
            rewards_buffer.extend(list(info.values()))

            memory_buffer.extend(
                zip(states, actions, rewards, next_states, dones))
            sample = memory_buffer.sample()
            if sample:
                states_batch, action_batch, reward_batch, \
                    next_states_batch, done_batch = sample

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
                summaries, total_t, _ = estimator.update(
                    sess, states_batch, action_batch, targets_batch)

                if total_t % update_target_every == 0:
                    estimator.target_update(sess)

                if total_t % save_model_every == 0:
                    saver.save(
                        sess,
                        os.path.join(checkpoint_path, 'model'),
                        total_t,
                        write_meta_graph=False)
                    print("Save session, global_step: {}.".format(total_t))

            states = next_states

            # Add summaries to tensorboard per 100 iterations
            if total_t % 100 == 0:
                summary_writer.add_summary(summaries, total_t)

                if rewards_buffer:
                    mean_reward = sum(rewards_buffer) / len(rewards_buffer)
                    episode_summary = tf.Summary()
                    episode_summary.value.add(
                        simple_value=mean_reward,
                        node_name="rewards",
                        tag="rewards")

                    summary_writer.add_summary(episode_summary, total_t)
                    summary_writer.flush()
                    rewards_buffer.clear()

    except KeyboardInterrupt:
        print("\nKeyboard interrupt!")

    except Exception:
        traceback.print_exc()

    finally:
        saver.save(
            sess,
            os.path.join(checkpoint_path, 'model'),
            total_t,
            write_meta_graph=False)
        env.close()

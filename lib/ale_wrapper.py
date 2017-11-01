import cv2
import numpy as np
from collections import deque
from ale_python_interface import ALEInterface


class Wrapper(object):
    def __init__(self, env, skip=4):
        self.env = env
        self.skip = skip
        self.action = env.getMinimalActionSet()
        self.obs_buffer = deque(maxlen=2)

    def step(self, a):
        total_reward = 0.0
        done = None
        action = self.action[a]
        for _ in range(self.skip):
            r = self.env.act(action)
            obs = Wrapper.process(self.env.getScreenGrayscale())
            done = self.env.game_over()
            total_reward += r
            self.obs_buffer.append(obs)
            if done:
                break
        max_frame = np.max(np.stack(self.obs_buffer), axis=0)
        return max_frame, total_reward, done

    def reset(self):
        self.obs_buffer.clear()
        self.env.reset_game()
        obs = Wrapper.process(self.env.getScreenGrayscale())
        self.obs_buffer.append(obs)
        return obs

    @staticmethod
    def process(obs):
        obs = np.array(obs)
        obs = cv2.resize(obs, (84, 110), interpolation=cv2.INTER_AREA)
        obs = obs[18:102, :]
        obs = np.reshape(obs, [84, 84, 1])
        return obs


class Framestack(object):
    def __init__(self, env, k):
        self.env = env
        self.k = k
        self.obs = deque(maxlen=k)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.obs.append(obs)
        return np.concatenate(self.obs, axis=2)

    def step(self, a):
        obs, reward, done = self.env.step(a)
        self.obs.append(obs)
        return np.concatenate(self.obs, axis=2), reward, done, None


def wrapper_env(game_name, skip=4, stack=4):
    env = ALEInterface()
    name = './games/' + game_name + '.bin'
    env.loadROM(name.encode('utf-8'))
    env = Wrapper(env, skip=skip)
    env = Framestack(env, stack)
    return env

import gym


def make_env(game_name):
    env = gym.make(game_name)
    return env

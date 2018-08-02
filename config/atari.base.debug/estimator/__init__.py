from .dqn import Dqn


def get_estimator(model_name, n_ac, lr, discount, **kwargs):
    if model_name == 'dqn':
        return Dqn(n_ac, lr=lr, discount=discount)

from .dqn import Dqn
from .dqn import DoubleDqn
from .dqn import DuelingNetwork
from .multitarget import AveDqn
from .distributional import DistributionalDqn


def get_estimator(model_name, n_ac, lr, discount, **kwargs):
    if model_name == 'dqn':
        return Dqn(n_ac, lr=lr, discount=discount)

    if model_name == 'doubledqn':
        return DoubleDqn(n_ac, lr=lr, discount=discount)

    if model_name == 'dueling':
        return DuelingNetwork(n_ac, lr=lr, discount=discount)

    if model_name.startswith('avedqn-'):
        k = int(model_name.split('-')[1])
        return AveDqn(n_ac, lr=lr, discount=discount, k=k)

    if model_name == 'c51':
        return DistributionalDqn(
            n_ac, lr=lr, discount=discount, vmax=10, vmin=-10, n_atoms=51)

    raise NotImplementedError

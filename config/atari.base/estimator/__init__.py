from .dqn import Dqn
from .dqn import SoftDqn
from .avedqn import AveDqn
from .distdqn import DistDqn
from .doubledqn import DoubleDqn


def get_estimator(model_name, n_ac, lr, discount, **kwargs):
    if model_name == 'dqn':
        estimator = Dqn(n_ac, lr=lr, discount=discount)
    elif model_name == 'softdqn':
        estimator = SoftDqn(n_ac, lr=lr, discount=discount, tau=kwargs['tau'])
    elif model_name == 'doubledqn':
        estimator = DoubleDqn(n_ac, lr=lr, discount=discount)
    elif model_name == 'distdqn':
        estimator = DistDqn(
            n_ac, lr=lr, discount=discount, vmax=10, vmin=-10, n_atoms=51)

    elif model_name.startswith('avedqn-'):
        k = int(model_name.split('-')[1])
        estimator = AveDqn(n_ac, lr=lr, discount=discount, k=k)

    else:
        raise Exception('{} is not supported!'.format(model_name))

    return estimator

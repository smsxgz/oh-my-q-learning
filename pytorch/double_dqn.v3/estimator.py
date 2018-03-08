from torch import nn
import torch.nn.functional as F
from torch import optim
from util import *


class Q_Net(nn.Module):
    def __init__(self, num_actions=18, activation=F.relu):
        super(Q_Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


class Estimator(object):
    def __init__(self, action_n, lr, update_target_rho=0.01):
        """
        Notes:
            if update_target_rho is 1, we will copy q's parameters to target's
            parameters, and we should set update_target_every to be larger
            like 1000.
        """
        self.activation_fn = F.relu
        self.update_target_rho = update_target_rho
        self.net = Q_Net(num_actions=action_n, activation=self.activation_fn)
        self.target_net = Q_Net(num_actions=action_n, activation=self.activation_fn)

        if use_cuda:
            self.net.cuda()
            self.target_net.cuda()

        self.target_update(hard=True)
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=lr)

    def target_update(self, hard=False):
        params = get_flat_params_from(self.net)
        params_target = get_flat_params_from(self.target_net)
        if hard:
            set_flat_params_to(self.target_net, params)
        else:
            new_params = self.update_target_rho * params + (1 - self.update_target_rho) * params_target
            set_flat_params_to(self.target_net, new_params)

    def predict(self, s, target=False):
        s = turn_into_cuda(np_to_var(s))
        if target:
            return self.target_net(s).cpu().data.numpy()
        else:
            return self.net(s).cpu().data.numpy()

    def update(self, s, a, y):
        s = turn_into_cuda(np_to_var(s))
        a = turn_into_cuda(np_to_var(a))
        y = turn_into_cuda(np_to_var(y)).view(-1, 1)

        y_pred = self.net(s).gather(1, a.long())

        explained_var = 1 - torch.var(y - y_pred) / torch.var(y)
        average_q = torch.mean(y_pred)

        loss = (y - y_pred).pow(2).mean()
        info = {'explained_var': explained_var.data[0], 'loss': loss.data[0], 'average_q': average_q.data[0]}

        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()

        return info


import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    var = Variable(
        torch.from_numpy(ndarray),
        volatile=volatile,
        requires_grad=requires_grad).type(dtype)
    return var.cuda() if USE_CUDA else var


class Q_Net(nn.Module):
    def __init__(self, action_n, activation):
        super(Q_Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(64 * 7 * 7, 512)
        self.fc5 = nn.Linear(512, action_n)
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
        self.net = Q_Net(action_n, F.relu)
        self.target_net = Q_Net(action_n, F.relu)

        if USE_CUDA:
            self.net.cuda()
            self.target_net.cuda()

        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)

    def predict(self, s):
        return to_numpy(self.net(to_tensor(s)))

    def update(self, discount_factor, states_batch, action_batch, reward_batch,
               next_states_batch, done_batch):
        q_values_next = self.net(to_tensor(next_states_batch))
        best_actions = q_values_next.max(1)

        q_values_next_target = self.target_net(
            to_tensor(next_states_batch, volatile=True))
        discount_factor_batch = discount_factor * to_tensor(
            np.invert(done_batch).astype(np.float32))
        targets_batch = to_tensor(reward_batch) + discount_factor_batch * \
            q_values_next_target.gather(1, best_actions.long())

        predictions = self.net(to_tensor(states_batch)).gather(
            1,
            to_tensor(action_batch).long())

        max_q_value = torch.max(predictions)
        min_q_value = torch.min(predictions)

        loss = (predictions - targets_batch).pow(2).mean()
        summaries = [loss.data[0], max_q_value.data[0], min_q_value[0]]

        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.tot += 1

        return

    def target_update(self):
        for e2_v, e1_v in zip(self.target_net.parameters(),
                              self.net.parameters()):
            e2_v.data.copy_(
                (e1_v.data - e2_v.data) * self.update_target_rho + e2_v.data)

    def save(self):
        pass

    def restore(self):
        pass

import os
import json
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
INT = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    var = Variable(
        torch.from_numpy(ndarray),
        volatile=volatile,
        requires_grad=requires_grad).type(dtype)
    return var.cuda() if USE_CUDA else var


class Q_Net(nn.Module):
    def __init__(self, state_shape, action_n, activation):
        assert list(state_shape) == [4, 84, 84]
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
    def __init__(self, state_shape, action_n, lr, update_target_rho=0.01):
        """
        Notes:
            if update_target_rho is 1, we will copy q's parameters to target's
            parameters, and we should set update_target_every to be larger
            like 1000.
        """
        self.activation_fn = F.relu
        self.update_target_rho = update_target_rho
        self.net = Q_Net(state_shape, action_n, self.activation_fn)
        self.target_net = Q_Net(state_shape, action_n, self.activation_fn)

        if USE_CUDA:
            self.net.cuda()
            self.target_net.cuda()

        self.total_t = 0
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)

    def predict(self, s):
        return to_numpy(self.net(to_tensor(s)))

    def update(self, discount_factor, states_batch, action_batch, reward_batch,
               next_states_batch, done_batch):
        batch_size = states_batch.shape[0]

        q_values_next = to_numpy(self.net(to_tensor(next_states_batch)))
        best_actions = np.argmax(q_values_next, axis=1)

        q_values_next_target = to_numpy(
            self.target_net(to_tensor(next_states_batch)))
        discount_factor_batch = discount_factor * np.invert(done_batch).astype(
            np.float32)
        targets_batch = reward_batch + discount_factor_batch * \
            q_values_next_target[np.arange(batch_size), best_actions]
        targets_batch = to_tensor(targets_batch).view(-1, 1)

        predictions = self.net(to_tensor(states_batch)).gather(
            1,
            to_tensor(action_batch, dtype=INT).view(-1, 1))

        max_q_value = torch.max(predictions)
        min_q_value = torch.min(predictions)

        loss = (predictions - targets_batch).pow(2).mean()
        summaries = [loss.data[0], max_q_value.data[0], min_q_value[0]]

        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_t += 1

        return self.total_t, summaries

    def target_update(self):
        for e2_v, e1_v in zip(self.target_net.parameters(),
                              self.net.parameters()):
            e2_v.data.copy_(
                (e1_v.data - e2_v.data) * self.update_target_rho + e2_v.data)

    def save(self, checkpoint_path):
        self.net.cpu()
        torch.save(self.net.state_dict(),
                   os.path.join(checkpoint_path, 'model-{}.pkl'.format(
                       self.total_t)))
        self.net.cuda()

        self.target_net.cpu()
        torch.save(self.target_net.state_dict(),
                   os.path.join(checkpoint_path, 'target_model-{}.pkl'.format(
                       self.total_t)))
        self.target_net.cuda()

        # save total_t
        checkpoints_json_path = os.path.join(checkpoint_path,
                                             'checkpoints.json')
        if os.path.exists(checkpoints_json_path):
            with open(checkpoints_json_path, 'r') as f:
                checkpoints = json.load(f)
        else:
            checkpoints = []

        # max_to_keep = 50
        if len(checkpoints) == 50:
            tot = checkpoints.pop(0)
            os.remove(
                os.path.join(checkpoint_path, 'model-{}.pkl'.format(tot)))
            os.remove(
                os.path.join(checkpoint_path,
                             'target_model-{}.pkl'.format(tot)))

        checkpoints.append(self.total_t)
        with open(checkpoints_json_path, 'w') as f:
            json.dump(checkpoints, f)

    def restore(self, checkpoint_path):
        checkpoints_json_path = os.path.join(checkpoint_path,
                                             'checkpoints.json')
        if os.path.exists(checkpoints_json_path):
            with open(checkpoints_json_path, 'r') as f:
                self.total_t = json.load(f)[-1]
            print('restore network from checkpoint {}...'.format(self.total_t))

            self.net.load_state_dict(
                torch.load(
                    os.path.join(checkpoint_path, 'model-{}.pkl'.format(
                        self.total_t))))
            self.target_net.load_state_dict(
                torch.load(
                    os.path.join(checkpoint_path, 'target_model-{}.pkl'.format(
                        self.total_t))))

        else:
            print('New start!!')

    def get_global_step(self):
        return self.total_t

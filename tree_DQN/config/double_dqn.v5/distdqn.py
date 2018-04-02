import os
import json
import torch
import numpy as np
from util import *
from network import Dist_Q_Net


class DistDQN(object):
    def __init__(self, state_shape, action_n, lr, update_target_rho=0.01, n_atoms=51, vmax=10, vmin=-10):
        self.n_ac = action_n
        self.n_atoms = n_atoms
        self.vmax = vmax
        self.vmin = vmin
        self.delta = (vmax - vmin) / (n_atoms - 1)
        self.split_points = np.linspace(vmin, vmax, n_atoms)

        # establish qnet and target qnet
        self.update_target_rho = update_target_rho
        self.net = Dist_Q_Net(state_shape, action_n, n_atoms)
        self.target_net = Dist_Q_Net(state_shape, action_n, n_atoms)

        if USE_CUDA:
            self.net.cuda()
            self.target_net.cuda()

        self.update_target()
        self.total_t = 0
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def get_action(self, obs):
        logits = self.net(to_tensor(obs))
        probs_tensor = torch.stack(list(map(nn.Softmax(dim=1), logits.chunk(self.n_ac, 1))), 1)
        probs = probs_tensor.cpu().data.numpy()
        qvals = np.sum(probs * self.split_points, axis=-1)
        return np.argmax(qvals, axis=1)

    def update(self, discount, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        batch_size = state_batch.shape[0]

        state_batch_tensor = to_tensor(state_batch)
        q_logits_tensor = self.net(state_batch_tensor)
        q_probs_tensor = torch.stack(list(map(nn.Softmax(dim=1), q_logits_tensor.chunk(self.n_ac, 1))), 1)
        # action batch should be np.int64!!!!!!
        action_batch = action_batch.astype(np.int64)
        q_probs_action_tensor = q_probs_tensor[np.arange(batch_size), action_batch]
        q_probs_action_tensor.data.clamp_(0.0001, 0.9999)

        next_state_batch_tensor = to_tensor(next_state_batch)
        next_q_logits_tensor = self.target_net(next_state_batch_tensor)
        q_probs_next = to_numpy(torch.stack(list(map(nn.Softmax(dim=1), next_q_logits_tensor.chunk(self.n_ac, 1))), 1))
        q_values_next = np.sum(q_probs_next * self.split_points, axis=-1)
        best_action = np.argmax(q_values_next, axis=1)

        targets = []
        for reward, probs, done in zip(reward_batch, q_probs_next[np.arange(batch_size), best_action], done_batch):
            targets.append(self._calc_dist(reward, discount * (1 - done), probs))
        targets_tensor = to_tensor(np.array(targets), requires_grad=False)

        self.net.zero_grad()
        entropy_loss = - torch.mean(torch.sum(targets_tensor * torch.log(q_probs_action_tensor), 1))
        entropy_loss.backward()
        self.optimizer.step()
        self.total_t += 1

        return self.total_t, {'loss': entropy_loss.data[0]}

    def update_target(self):
        for e2_v, e1_v in zip(self.target_net.parameters(), self.net.parameters()):
            e2_v.data.copy_((e1_v.data - e2_v.data) * self.update_target_rho + e2_v.data)

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
        checkpoints_json_path = os.path.join(checkpoint_path, 'checkpoints.json')
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

    def _interception(self, x):
        if x > self.vmax:
            return self.vmax
        elif x < self.vmin:
            return self.vmin
        else:
            return x

    def _calc_dist(self, reward, discount, probs):
        m = np.zeros(self.n_atoms, dtype=np.float32)
        for (p, z) in zip(probs, self.split_points):
            projection = self._interception(reward + discount * z)
            b = (projection - self.vmin) / self.delta
            l = int(b)
            m[l] += p * (1 + l - b)
            if l < self.n_atoms - 1:
                m[l + 1] += p * (b - l)
        return m

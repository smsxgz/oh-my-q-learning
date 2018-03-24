import os
import json
import torch
import numpy as np
from util import *
from network import Q_Net
from torch.nn import functional as F


class DoubleDQN(object):
    def __init__(self, state_shape, action_n, lr, update_target_rho=0.01, N_leaf=20):
        """
        Notes:
            if update_target_rho is 1, we will copy q's parameters to target's
            parameters, and we should set update_target_every to be larger
            like 1000.
        """
        self.update_target_rho = update_target_rho
        self.net = Q_Net(state_shape, N_leaf)
        self.target_net = Q_Net(state_shape, N_leaf)
        self.N_leaf = N_leaf
        self.action_n = action_n
        self.leaf = np.random.randn(N_leaf * action_n).reshape((N_leaf, action_n))
        self.target_leaf = np.random.randn(N_leaf * action_n).reshape((N_leaf, action_n))

        if USE_CUDA:
            self.net.cuda()
            self.target_net.cuda()

        self.update_target()
        self.total_t = 0
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)

    def get_action(self, obs):
        probs = to_numpy(self.net(to_tensor(obs)))
        q_values = np.dot(probs, self.leaf)
        return np.argmax(q_values, axis=1)

    def update(self, discount, state_batch, action_batch, reward_batch,
               next_state_batch, done_batch):
        batch_size = state_batch.shape[0]

        probs_next = to_numpy(self.net(to_tensor(next_state_batch)))
        q_values_next = np.dot(probs_next, self.leaf)
        best_actions = np.argmax(q_values_next, axis=1)

        probs_next_target = to_numpy(self.target_net(to_tensor(next_state_batch)))
        q_values_next_target = np.dot(probs_next_target, self.target_leaf)
        discount_factor_batch = discount * np.invert(done_batch).astype(np.float32)
        targets_batch = reward_batch + discount_factor_batch * q_values_next_target[np.arange(batch_size), best_actions]

        squ = [np.square(self.leaf[:, action_batch[i]] - targets_batch[i]) for i in range(batch_size)]
        squ = to_tensor(np.array(squ)) / batch_size

        predictions = self.net(to_tensor(state_batch))
        loss = torch.sum(squ * predictions)
        summaries = {'loss': loss.data[0], 'max_prob': np.mean(np.max(to_numpy(predictions), axis=1))}

        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_t += 1

        return self.total_t, summaries

    def update_leaf(self, discount, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        batch_size = state_batch.shape[0]

        probs_next = to_numpy(self.net(to_tensor(next_state_batch, volatile=True)))
        q_values_next = np.dot(probs_next, self.leaf)
        best_actions = np.argmax(q_values_next, axis=1)

        probs_next_target = to_numpy(self.target_net(to_tensor(next_state_batch, volatile=True)))

        q_values_next_target = np.dot(probs_next_target, self.target_leaf)
        discount_factor_batch = discount * np.invert(done_batch).astype(np.float32)
        targets_batch = reward_batch + discount_factor_batch * q_values_next_target[np.arange(batch_size), best_actions]
        targets_batch = np.reshape(targets_batch, (-1, 1))

        predictions = to_numpy(self.net(to_tensor(state_batch, volatile=True)))
        comb = targets_batch * predictions

        for k in range(self.action_n):
            sele_comb = comb[action_batch == k]
            sele_pred = predictions[action_batch == k]
            if len(sele_comb) > 0:
                self.leaf[:, k] = self.leaf[:, k] * 0.99 + 0.01 * np.sum(sele_comb, axis=0) / np.sum(sele_pred, axis=0)

    def update_target(self):
        for e2_v, e1_v in zip(self.target_net.parameters(),
                              self.net.parameters()):
            e2_v.data.copy_(
                (e1_v.data - e2_v.data) * self.update_target_rho + e2_v.data)
        self.target_leaf = self.leaf.copy()

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
        checkpoints_json_path = os.path.join(checkpoint_path, 'checkpoints.json')
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

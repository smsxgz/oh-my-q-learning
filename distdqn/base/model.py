import numpy as np
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from network import qnet

class Distdqn(object):
    def __init__(self, n_ac, n_atoms, discount=0.98, epsilon=0.05, vmax=10, vmin=-10, lr=1e-4, use_cuda=False):
        self.n_ac = n_ac
        self.n_atoms = n_atoms
        self.discount = discount
        self.epsilon = epsilon
        self.vmax = vmax
        self.vmin = vmin
        self.delta = (vmax - vmin) / (n_atoms - 1)
        self.split_points = np.linspace(vmin, vmax, n_atoms)
        self.use_cuda = use_cuda
        
        # establish qnet and target qnet
        self.qdist = qnet(n_ac, n_atoms)
        self.qdist_target = qnet(n_ac, n_atoms)
        self.optimizer = Adam(self.qdist.parameters(), lr=lr)

        # init qnet and target qnet
        self.update_target()
        if self.use_cuda:
            self.float = torch.cuda.FloatTensor
            self.cuda()
        else:
            self.float = torch.FloatTensor
    
    def interception(self, x):
        if x > self.vmax:
            return self.vmax
        elif x < self.vmin:
            return self.vmin
        else:
            return x

    def calc_dist(self, reward, discount, probs):
        m = np.zeros(self.n_atoms, dtype=np.float32)
        for (p, z) in zip(probs, self.split_points):
            projection = self.interception(reward + discount * z)
            b = (projection - self.vmin) / self.delta
            l = int(b)
            m[l] += p * (1 + l - b)
            if l < self.n_atoms - 1:
                m[l + 1] += p * (b - l)
        return m

    def update_target(self):
        for param_target, param in zip(self.qdist_target.parameters(), self.qdist.parameters()):
            param_target.data.copy_(param.data)

    def get_action(self, obs):
        x = Variable(torch.from_numpy(obs)).type(self.float)
        logits = self.qdist(x)
        probs_tensor = torch.stack(list(map(nn.Softmax(), logits.chunk(self.n_ac, 1))), 1)
        if self.use_cuda:
            probs = probs_tensor.cpu().data.numpy()
        else:
            probs = probs_tensor.data.numpy()
        qvals = np.sum(probs * self.split_points, axis=-1)
        best_action = np.argmax(qvals, axis=1)
        batch_size = obs.shape[0]
        actions = np.random.randint(self.n_ac, size=batch_size)
        idx = np.random.uniform(size=batch_size) > self.epsilon
        actions[idx] = best_action[idx]
        return best_action

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        batch_size = state_batch.shape[0]
        state_batch_tensor = Variable(torch.from_numpy(state_batch)).type(self.float)
        q_logits_tensor = self.qdist(state_batch_tensor)
        q_probs_tensor = torch.stack(list(map(nn.Softmax(), q_logits_tensor.chunk(self.n_ac, 1))), 1)
        ## action batch should be np.int64!!!!!!
        action_batch = action_batch.astype(np.int64)
        q_probs_action_tensor = q_probs_tensor[np.arange(batch_size), action_batch]

        next_state_batch_tensor = Variable(torch.from_numpy(next_state_batch)).type(self.float)
        next_q_logits_tensor = self.qdist(next_state_batch_tensor)
        next_q_probs_tensor = torch.stack(list(map(nn.Softmax(), next_q_logits_tensor.chunk(self.n_ac, 1))), 1)
        if self.use_cuda:
            next_q_probs = next_q_probs_tensor.cpu().data.numpy()
        else:
            next_q_probs = next_q_probs_tensor.data.numpy()
        next_q_vals = np.sum(next_q_probs * self.split_points, axis=-1)
        best_action = np.argmax(next_q_vals, axis=1)
        
        targets = []
        for reward, probs, done in zip(reward_batch, next_q_probs[np.arange(batch_size), best_action], done_batch):
            targets.append(self.calc_dist(reward, self.discount * (1 - done), probs))
        targets = np.array(targets)
        targets_tensor = Variable(torch.from_numpy(targets)).type(self.float)
        self.qdist.zero_grad()
        entropy_loss = - torch.mean(torch.sum(targets_tensor * torch.log(q_probs_action_tensor), 1))
        entropy_loss.backward()
        self.optimizer.step()
        return entropy_loss
    
    def eval(self):
        # for BN
        self.qdist.eval()
        self.qdist_target.eval()

    def train(self):
        # for BN
        self.qdist.train()
        self.qdist_target.train()

    def cuda(self):
        self.qdist.cuda()
        self.qdist_target.cuda()

    def save_model(self, outdir, step):
        modelnames = os.listdir(outdir)
        nums = []
        for modelname in modelnames:
            if 'qdist' in modelname:
                num = modelname[5:-4]
                nums.append(int(num))
        if len(nums) == 10:
            os.remove('{}/qdist{}.pkl'.format(outdir, min(nums)))
        
        if self.use_cuda:
            self.qdist.cpu()
        torch.save(self.qdist.state_dict(), '{}/qdist{}.pkl'.format(outdir, step))
        if self.use_cuda:
            self.qdist.cuda()

    def load_model(self, outdir):
        modelnames = os.listdir(outdir)
        max_num = 0
        for modelname in modelnames:
            if 'qdist' in modelname:
                num = modelname[5:-4]
                if int(num) > max_num:
                    max_num = int(num)
        self.qdist.load_state_dict(torch.load('{}/qdist{}.pkl'.format(outdir, max_num)))
        self.qdist_target.load_state_dict(torch.load('{}/qdist{}.pkl'.format(outdir, max_num)))

        
if __name__ == '__main__':
    a = Distdqn(3, 10, 50)

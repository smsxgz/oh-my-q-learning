import torch
from torch.optim.optimizer import Optimizer
from scipy.stats import truncnorm
from torch.autograd import Variable


class Anderson(Optimizer):
    def __init__(self, params, lr=0.1, weight_decay=0):
        defaults = dict(lr=lr, alpha1=0.9, weight_decay=weight_decay)
        super(Anderson, self).__init__(params, defaults)

        self.prev = []
        for group in self.param_groups:
            tmp_group = {'params': []}
            for p in group['params']:
                q = p.clone()
                q.data = torch.from_numpy(
                    truncnorm.rvs(-2, 2, size=p.size()).astype('float32') /
                    100).cuda()
                q.grad = Variable(
                    torch.from_numpy(
                        truncnorm.rvs(-2, 2, size=p.size()).astype('float32') /
                        100).cuda())
                tmp_group['params'].append(q)
            self.prev.append(tmp_group)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        a = 0
        b = 0
        for group1, group2 in zip(self.param_groups, self.prev):
            for p, q in zip(group1['params'], group2['params']):
                if p.grad is None:
                    continue
                tmp = q.grad.data - p.grad.data
                a += (tmp * q.grad.data).sum()
                b += (tmp * tmp).sum()
        alpha1 = a / b
        alpha2 = 1 - alpha1

        for group1, group2 in zip(self.param_groups, self.prev):

            beta = group1['lr']
            weight_decay = 1 - group1['weight_decay'] * beta

            p = group1['params']
            q = group2['params']

            for i in range(len(group1['params'])):
                if p[i].grad is None:
                    continue
                tmp = p[i].clone()

                p[i].data = alpha1 * weight_decay * p[i].data + \
                    alpha2 * weight_decay * q[i].data - \
                    beta * alpha1 * p[i].grad.data - \
                    beta * alpha2 * q[i].grad.data
                q[i] = tmp
                q[i].grad = p[i].grad.clone()

        return loss


class Interpolation(Optimizer):
    def __init__(self, params, lr=0.1, alpha1=0.1, weight_decay=0):
        defaults = dict(lr=lr, alpha1=0.9, weight_decay=weight_decay)
        super(Interpolation, self).__init__(params, defaults)

        self.prev = []
        for group in self.param_groups:
            tmp_group = {'params': []}
            for p in group['params']:
                q = p.clone()
                q.data = torch.from_numpy(
                    truncnorm.rvs(-2, 2, size=p.size()).astype('float32') /
                    100).cuda()
                q.grad = Variable(
                    torch.from_numpy(
                        truncnorm.rvs(-2, 2, size=p.size()).astype('float32') /
                        100).cuda())
                tmp_group['params'].append(q)
            self.prev.append(tmp_group)

        self.alpha1 = alpha1
        self.alpha2 = 1 - alpha1

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group1, group2 in zip(self.param_groups, self.prev):
            beta = group1['lr']
            weight_decay = 1 - group1['weight_decay'] * beta

            p = group1['params']
            q = group2['params']

            for i in range(len(group1['params'])):
                if p[i].grad is None:
                    continue
                tmp = p[i].clone()

                p[i].data = self.alpha1 * weight_decay * p[i].data + \
                    self.alpha2 * weight_decay * q[i].data - \
                    beta * self.alpha1 * p[i].grad.data - \
                    beta * self.alpha2 * q[i].grad.data
                q[i] = tmp
                q[i].grad = p[i].grad.clone()
        return loss

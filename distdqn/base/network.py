import torch
import torch.nn as nn

class qnet(nn.Module):
    def __init__(self, nb_actions, nb_atoms):
        super(qnet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, nb_actions * nb_atoms)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)   
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    import numpy as np
    from torch.autograd import Variable
    a = np.arange(10)
    from IPython import embed; embed()

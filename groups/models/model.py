import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import random
from easyfl.models.meta_layers import *


class WNet(nn.Module):
    def __init__(self, input, hidden, output):
        super(WNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden, output)

    def forward(self, x):

        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)


class WNet_1(MetaModule):
    def __init__(self, input, hidden, output):
        super(WNet_1, self).__init__()
        self.linear1 = MetaLinear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)

        out = self.linear2(x)
        return torch.sigmoid(out)
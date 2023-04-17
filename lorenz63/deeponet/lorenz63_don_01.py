import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs


class PIDeepONet(nn.Module):
    def __init__(self, trunk_layers, branch_layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_func = nn.MSELoss(reduction='mean')
        self.trunk_linear = nn.ModuleList(
            [nn.Linear(trunk_layers[i], trunk_layers[i + 1]) for i in range(len(trunk_layers) - 1)])
        self.branch_linear = nn.ModuleList(
            [nn.Linear(branch_layers[i], branch_layers[i + 1]) for i in range(len(branch_layers) - 1)])

    def forward(self, x):
        pass

    def loss_bc(self, x_bc, y_bc):
        pass

    def loss_pde(self, x_pde):
        pass

    def loss(self, x_bc, y_bc, x_pde):
        pass

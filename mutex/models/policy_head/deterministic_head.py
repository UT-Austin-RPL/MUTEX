import torch
import torch.nn as nn


class DeterministicHead(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size=1024,
                 num_layers=2):

        super().__init__()
        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i+1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]

        if self.action_squash:
            layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y

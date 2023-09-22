import torch.nn as nn
from mutex.utils import *

class MLPEncoder(nn.Module):
    """ Encode task embedding

        h = f(e), where
            e: pretrained task embedding from large model
            h: latent embedding (B, H)
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers):
        super().__init__()
        assert num_layers >= 1, "[error] num_layers < 1"
        sizes = [input_size] + [hidden_size] * (num_layers-1) + [output_size]
        layers = []
        for i in range(num_layers-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.projection = nn.Sequential(*layers)

    def forward(self, data):
        """
        data:
            task_emb: (B, E)
        """
        h = self.projection(data["task_emb"]) # (B, L, H)
        return h

import torch.nn as nn
from mutex.utils import *


class IdentityEncoder(nn.Module):
    """ Dummy encoder that directly outputs the pretrained task embedding
        h = e
            e: pretrained task embedding (B, E)
    """

    def __init__(self, dummy=True):
        super().__init__()

    def forward(self, data):
        """
        data:
            task_emb: (B, E)
        """
        h = data["task_emb"] # (B, L, H)
        return h

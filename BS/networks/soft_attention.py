import torch.nn as nn


class Soft_attention(nn.Module):
    def __init__(self, in_channels, relation_vec):
        super(Soft_attention, self).__init__()
        self.in_channels = in_channels
        self.relation_vec = relation_vec

    
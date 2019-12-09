import torch.nn as nn


class SoftAttention(nn.Module):
    def __init__(self, in_channels, compare_vec):
        super(SoftAttention, self).__init__()
        self.in_channels = in_channels
        self.relation_vec = compare_vec

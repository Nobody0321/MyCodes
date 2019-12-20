import torch
import torch.nn as nn
import math
from enum import IntEnum


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        """
        for translation, mask is necessary, but for relation extraction, it"s not.
        q, k, v: (n, l, d_feature)
        """
        d_k = k.size(-1)  # get the size of the key
        assert q.size(-1) == d_k  # otherwise the 2 metrics cannot be multiplied

        attn = torch.bmm(q, k.transpose(1, 2))  # (n, l, l), a look up matrix
        attn = attn / math.sqrt(d_k)
        # print("attn", attn)
        attn = torch.exp(attn)
        attn = attn / attn.sum(-1, keepdim=True)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)  # (n, l, d_feature)
        return output


class AttentionHead(nn.Module):
    def __init__(self, d_model, d_feature, dropout=0.1):
        super().__init__()
        # We will assume the queries, keys, and values all have the same feature size
        self.attn = ScaledDotProductAttention(dropout)
        self.query_transform = nn.Linear(d_model, d_feature)
        self.key_transform = nn.Linear(d_model, d_feature)
        self.value_transform = nn.Linear(d_model, d_feature)

    def forward(self, queries, keys, values):
        # (n, l, d_model) -> (n, l, d_feature)
        queries = self.query_transform(queries)
        keys = self.key_transform(keys)
        values = self.value_transform(values)
        # compute multiple attention weighted sums
        x = self.attn(queries, keys, values)  # (n, l, d_feature)
        return x


class MultiHeadAttention(nn.Module):
    """The complete multi-head attention block"""

    def __init__(self, d_model, d_feature, n_heads, dropout=0.1):
        """
        d_model: dim of the blstm output
        d_feature: dim of the Q, K, V
        """
        super().__init__()
        # in practice, d_model == d_feature * n_heads
        # print(d_model, d_feature, n_heads)
        assert d_model == d_feature * n_heads

        self.d_model = d_model
        self.d_feature = d_feature
        self.n_heads = n_heads

        self.attn_heads = nn.ModuleList([
            AttentionHead(d_model, d_feature, dropout) for _ in range(n_heads)
        ])
        self.projection = nn.Linear(d_feature * n_heads, d_model)

    def forward(self, queries, keys, values):
        """
        queries, keys, values: (n, l, d_model)
        """
        x = [attn(queries, keys, values)  # n_heads * (n, l, d_feature)
             for i, attn in enumerate(self.attn_heads)]

        # concatenate again
        x = torch.cat(x, dim=Dim.feature)  # (n, l, d_feature * n_heads)
        x = self.projection(x)  # (n, l, d_model)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, d_feature=64,
                 d_ff=2048, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x : (n, l, d_model)
        """
        att = self.attn_head(x, x, x)  # self attention (n, l, d_model) -> (n, l, d_model)
        # Apply normalization and residual connection
        x = x + self.dropout(self.layer_norm1(att))  # (n, l, d_model) -> (n, l, d_model)
        # Apply position-wise feed-forward networks
        pos = self.position_wise_feed_forward(x)  # (n, l, d_model) -> (n, l, d_ff) -> (n, l, d_model)
        # Apply normalization and residual connection
        x = x + self.dropout(self.layer_norm2(pos))  # (n, l, d_model)
        return x  # (n, l, d_model)

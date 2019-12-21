import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
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
        attn = torch.exp(attn)
        attn = attn / attn.sum(-1, keepdim=True)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)  # (n, l, d_feature)
        return output


class AttentionHead(nn.Module):
    def __init__(self, d_model, d_feature, dropout):
        super().__init__()
        # We will assume the queries, keys, and values all have the same feature size
        self.attn = ScaledDotProductAttention(dropout)
        self.query_transform = nn.Linear(d_model, d_feature)  # (l, 60) -> (l, 12)
        self.key_transform = nn.Linear(d_model, d_feature)
        self.value_transform = nn.Linear(d_model, d_feature)

    def init(self):
        nn.init.xavier_normal_(self.query_transform.weight.data)
        nn.init.xavier_normal_(self.query_transform.bias.data)
        nn.init.xavier_normal_(self.key_transform.weight.data)
        nn.init.xavier_normal_(self.key_transform.bias.data)
        nn.init.xavier_normal_(self.value_transform.weight.data)
        nn.init.xavier_normal_(self.value_transform.bias.data)

    def forward(self, queries, keys, values):
        # (n, l, 60) -> (n, l, 12)
        queries = self.query_transform(queries)
        keys = self.key_transform(keys)
        values = self.value_transform(values)
        # compute multiple attention weighted sums
        x = self.attn(queries, keys, values)  # (n, l, 12)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_output, dropout):
        """
        d_model: dim of the input
        d_feature: dim of the Q, K, V
        """
        super().__init__()

        self.d_model = d_model
        self.d_feature = d_model // n_heads
        self.n_heads = n_heads

        self.attn_heads = nn.ModuleList([
            AttentionHead(self.d_model, self.d_feature, dropout) for _ in range(n_heads)
        ])
        self.projection = nn.Linear(self.d_feature * n_heads, d_output)

    def forward(self, queries, keys, values):
        """
        queries, keys, values: (n, l, d_model)
        """
        x = [attn(queries, keys, values)  # n_heads * (n, l, d_feature)
             for i, attn in enumerate(self.attn_heads)]

        # concatenate again
        x = torch.cat(x, dim=2)  # (n, l, 60)
        x = self.projection(x)  # (n, l, 60) -> (n, l, 230)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, d_output, dropout):
        super(EncoderBlock, self).__init__()
        self.attn_head = MultiHeadAttention(d_model, n_heads, d_output, dropout)
        self.layer_norm1 = nn.LayerNorm(d_output, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(d_output, d_ff),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_output),
        )
        self.layer_norm2 = nn.LayerNorm(d_output, eps=1e-6)

    def forward(self, x):
        """
        input_x : (n, l, d_model)
        """
        x = self.attn_head(x, x, x)  # self attention (n, l, 60) -> (n, l, 230)
        # Apply normalization and residual connection
        x = x + self.dropout(self.layer_norm1(x))  # (n, l, 230) -> (n, l, 230)
        # Apply position-wise feed-forward networks
        pos = self.position_wise_feed_forward(x)  # (n, l, 230) -> (n, l, d_ff) -> (n, l, 230)
        # Apply normalization and residual connection
        x = x + self.dropout(self.layer_norm2(pos))  # (n, l, 230)
        return x  # (n, l, 230)

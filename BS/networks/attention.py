import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def _get_sinusoid_encoding_table(n_position, d_hid):
    """
    Sinusoid position encoding table
    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.Tensor(sinusoid_table).unsqueeze(0)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)
         
    def forward(self, x):
        return x + self.weight[:, :x.size(1), :]


class AttEncoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, config):
        super(AttEncoder, self).__init__()
        self.position_enc = PositionalEmbedding(config.att_d_input, max_len=config.max_len)
        self.dropout = nn.Dropout(p=config.att_drop_prob)
        self.layer_stack = nn.ModuleList([
            EncoderBlock(config.att_d_input, config.att_d_inner, config.att_d_ff, config.att_n_head, dropout=config.att_drop_prob)
            for _ in range(config.att_n_blocks)])
        self.layer_norm = nn.LayerNorm(config.att_d_model, eps=1e-6)

    def forward(self, src_seq):
        enc_output += self.position_enc(src_seq)
        enc_output = self.dropout(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)

        enc_output = self.layer_norm(enc_output)

        return enc_output


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_inner, d_ff, n_head, dropout=0.1):
        super().__init__()
        self.attn_head = MultiHeadAttention(d_model, d_inner, n_head, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
         
    def forward(self, x):
        att = self.attn_head(x, x, x)
        x = x + self.dropout(self.layer_norm1(att))
        pos = self.position_wise_feed_forward(x)
        x = x + self.dropout(self.layer_norm2(pos))
        return x


class MultiHeadAttention(nn.Module):
    """The full multihead attention block"""
    def __init__(self, d_model, d_inner, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
 
        self.attn_head = nn.ModuleList([
            AttentionHead(d_model, d_inner, dropout) for _ in range(n_head)
        ])
        self.projection = nn.Linear(d_inner * n_head, d_model) 
     
    def forward(self, queries, keys, values):
        x = [attn(queries, keys, values) # (Batch, Seq, Feature)
             for i, attn in enumerate(self.attn_head)]
         
        x = torch.cat(x, 2) # (Batch, Seq, d_inner * n_head)
        x = self.projection(x) # (Batch, Seq, D_Model)
        return x


class AttentionHead(nn.Module):
    """A single attention head"""
    def __init__(self, d_model, d_inner, dropout=0.1):
        super().__init__()
        # We will assume the queries, keys, and values all have the same feature size
        self.attn = ScaledDotProductAttention(dropout)
        self.query_tfm = nn.Linear(d_model, d_inner)
        self.key_tfm = nn.Linear(d_model, d_inner)
        self.value_tfm = nn.Linear(d_model, d_inner)
 
    def forward(self, queries, keys, values):
        Q = self.query_tfm(queries) # (Batch, Seq, Feature)
        K = self.key_tfm(keys) # (Batch, Seq, Feature)
        V = self.value_tfm(values) # (Batch, Seq, Feature)
        x = self.attn(Q, K, V)
        return x

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, q, k, v):
        d_k = k.size(-1) # get the size of the key 
        attn = F.softmax(torch.bmm(q / d_k**0.5, k.transpose(1, 2)), dim = 2)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v) 
        return output

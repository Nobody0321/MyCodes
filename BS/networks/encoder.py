from networks.embedding import *
from .attention import *
import torch.nn.functional as F
import torch.nn as nn


class _CNN(nn.Module):
    def __init__(self, config):
        super(_CNN, self).__init__()
        self.config = config
        self.in_channels = 1
        self.in_height = self.config.max_sen_length
        self.in_width = self.config.input_dim
        self.kernel_size = (self.config.window_size, self.in_width)
        self.out_channels = self.config.hidden_size
        self.stride = (1, 1)
        self.padding = (1, 0)
        self.cnn = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)

    def forward(self, embedding):
        return self.cnn(embedding)


class _PiecewisePooling(nn.Module):
    def __init(self):
        super(_PiecewisePooling, self).__init__()

    def forward(self, x, mask, hidden_size):
        mask = torch.unsqueeze(mask, 1)
        x, _ = torch.max(mask + x, dim=2)
        x = x - 100
        return x.view(-1, hidden_size * 3)


class _MaxPooling(nn.Module):
    def __init__(self):
        super(_MaxPooling, self).__init__()

    def forward(self, x, hidden_size):
        x, _ = torch.max(x, dim=1)
        return x.view(-1, hidden_size)


class PCNN(nn.Module):
    def __init__(self, config):
        super(PCNN, self).__init__()
        self.config = config
        self.mask = None
        self.cnn = _CNN(config)
        self.pooling = _PiecewisePooling()
        self.activation = nn.ReLU()

    def forward(self, embedding):
        embedding = torch.unsqueeze(embedding, dim=1)
        x = self.cnn(embedding)
        x = self.pooling(x, self.mask, self.config.hidden_size)
        return self.activation(x)


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        self.cnn = _CNN(config)
        self.pooling = _MaxPooling()
        self.activation = nn.ReLU()

    def forward(self, embedding):
        embedding = torch.unsqueeze(embedding, dim=1)
        x = self.cnn(embedding)
        x = self.pooling(x, self.config.hidden_size)
        return self.activation(x)


class soft_attention(nn.Module):
    def __init__(self, config):
        super(soft_attention, self).__init__()
        self.config = config
        self.activation = torch.tanh
        self.linear = torch.rand(1, config.hidden_dim).cuda()
        # nn.init.xavier_uniform_(self.linear.data)

    def forward(self, x):
        """

        :param x: (n, l, d)
        :return:
        """
        ret = []
        for each in x:
            M = torch.matmul(self.linear, self.activation(each.transpose(0, 1)))
            attention_weights = F.softmax(M, dim=1)  #
            each = torch.matmul(attention_weights, each)
            ret.append(self.activation(each.squeeze(0)))
        return torch.stack(ret)


# in pyTorch BiGru takes a 3-dim tensor as input, where the first dim represents the sentence_len,
# the second dim represents the batch_size,, and the third dim represents the word embedding_size
class BiGru(nn.Module):
    def __init__(self, config):
        super(BiGru, self).__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.sentence_len = config.sentence_len  # 120
        self.dropout = config.attn_dropout  # 0.1
        self.input_dim = config.input_dim  #
        self.out_channels = config.output_dim // 2  # hidden dim // 2 = 256 // 2 = 256
        self.rnn = nn.GRU(input_size=self.input_dim, hidden_size=self.out_channels, bidirectional=True)
        self.attn = SelfAttEncoder(config)

    def init_hidden(self):
        if self.config.use_gpu:
            return torch.stack([torch.zeros(2, self.sentence_len, self.out_channels).cuda(),
                                torch.zeros(2, self.sentence_len, self.out_channels).cuda()])
        else:
            return torch.stack([torch.zeros(2, self.sentence_len, self.out_channels),
                                torch.zeros(2, self.sentence_len, self.out_channels)])

    def forward(self, embedding):
        """
        :param embedding: word embeddings sentences, (n=batch_size/sen_num, sen_len=120, embedding_size=60)
        """
        embedding = embedding.permute(1, 0, 2)  # (sen_len=120, n, embedding_size=60)
        final_all, final_hidden = self.rnn(embedding)  # gtu_out(120, n, 256) stores the final state for all
        del final_all
        final_hidden = final_hidden.view(1, -1, self.config.out_channels)  # (2, sen_num, 128) -> (1, sen_num, 256)
        return final_hidden


class SelfAttention(nn.Module):
    def __init__(self, config, input_dim, output_dim=None):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoders = nn.ModuleList(
            [
                EncoderBlock(d_model=input_dim, n_heads=config.n_heads,
                             d_ff=config.d_ff, d_output=output_dim, dropout=config.attn_dropout)
                for _ in range(config.n_blocks)
            ]
        )

    def forward(self, x):
        """
        x : output of last layer, (n=batch_size, l=sen_len, 60)
        """
        for encoder in self.encoders:
            x = encoder(x)
        return x  # (n, l, 230)


class SentenceAtt(nn.Module):
    def __init__(self, word_dim):
        super(SentenceAtt, self).__init__()
        self.linear = nn.Linear(in_features=word_dim, out_features=1, bias=False)  # (230, 1)
        self.init()

    def init(self):
        nn.init.uniform_(self.linear.weight.data)

    def forward(self, sentence_vec):
        """

        :param sentence_vec: (120, 230)
        :return:
        """
        sentence_vec = torch.tanh(sentence_vec)  # (120, 230)
        attention_weights = F.softmax(self.linear(sentence_vec), dim=0)  # (120, 230) * 230, 1  = (120, 1)
        sentence_vec = torch.matmul(torch.transpose(attention_weights, 0, 1), sentence_vec)  # (1, 120) * (120,230)
        sentence_vec = torch.tanh(sentence_vec)
        return sentence_vec


class SelfAttEncoder(nn.Module):
    def __init__(self, config, input_dim, output_dim=None):
        super(SelfAttEncoder, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.attn_encoder = SelfAttention(config, self.input_dim, self.output_dim)
        # self.pooling = SentenceAtt(self.config.hidden_dim)

    def forward(self, x):
        """
        x : output of last layer, (n=batch_size, l=sen_len, 60)
        """
        x = self.attn_encoder(x)  # (n, l, 230)
        # perform max pooling in one sentence
        x = torch.max(x, dim=1)[0]  # (n, l, 230) -> (n, 230)
        # x = torch.stack([self.pooling(each).squeeze() for each in x])  # (n, l, 230) -> (n, 230)
        return x  # (n, 230)


class SelfSoftAttEncoder(nn.Module):
    def __init__(self, config, input_dim, output_dim=None):
        super(SelfSoftAttEncoder, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.attn_encoder = SelfAttention(config, self.input_dim, self.output_dim)
        self.pooling = SentenceAtt(self.config.hidden_dim)

    def forward(self, x):
        """
        x : output of last layer, (n=batch_size, l=sen_len, 60)
        """
        x = self.attn_encoder(x)  # (n, l, 230)
        x = torch.stack([self.pooling(each).squeeze() for each in x])  # (n, l, 230) -> (n, 230)
        return x  # (n, 230)

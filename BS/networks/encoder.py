import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from networks.attention import AttEncoderBlock


class _CNN(nn.Module):
    def __init__(self, config):
        super(_CNN, self).__init__()
        self.config = config
        self.in_channels = 1
        self.in_height = self.config.max_length
        self.in_width = self.config.word_size + 2 * self.config.pos_size
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


class SelfAttEncoderWithMax(nn.Module):
    def __init__(self, config):
        super(SelfAttEncoderWithMax, self).__init__()
        self.config = config
        self.attn_encoder = AttEncoderBlock(d_model=config.input_dim, n_heads=config.n_attn_heads, d_output=config.encoder_output_dim, dropout=config.attn_dropout)

    def forward(self, embedding):
        """

        :param embedding:
        :return:
        """
        x = self.attn_encoder(embedding)
        # perform max pooling
        x = torch.max(x, dim=1)[0]
        return x


class SelfAttEncoder(nn.Module):
    def __init__(self, config):
        super(SelfAttEncoder, self).__init__()
        self.config = config
        self.attn_encoder = AttEncoderBlock(d_model=config.input_dim, n_heads=config.n_attn_heads, d_output=config.hidden_size, dropout=config.attn_dropout)

    def forward(self, embedding):
        """

        :param embedding: n， 120, 65
        :return:
        """
        x = self.attn_encoder(embedding)  # n, 120, 230
        # perform max pooling
        return x


class BiGru_Att(nn.Module):
    def __init__(self, config):
        super(BiGru_Att, self).__init__()
        self.config = config
        self.rnn = nn.GRU(input_size=config.input_dim, hidden_size=config.encoder_output_dim//2, bidirectional=True)
        self.hidden = nn.Parameter(torch.zeros(2, config.max_length, config.encoder_output_dim//2))
        self.attn = SelfAttEncoderWithMax(config)

    def forward(self, x):
        self.rnn.flatten_parameters()
        output, _ = self.rnn(x, self.hidden)
        return self.attn(output)  # n, 120 ,23, -> n, 230


class SelfPCNN(nn.Module):
    def __init__(self, config):
        super(SelfPCNN, self).__init__()
        self.attn = AttEncoderBlock(d_model=config.input_dim, n_heads=config.n_attn_heads, d_output=config.input_dim, dropout=config.attn_dropout)

        self.PCNN = PCNN(config)

    def forward(self, embedding):
        embedding = self.attn(embedding)
        return self.PCNN(embedding)
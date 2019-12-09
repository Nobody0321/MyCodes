from networks.embedding import *
from models.attention import TransformerEncoder


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
        # mask = torch.unsqueeze(mask, 1)
        x, _ = torch.max(x, dim=2)
        x = x - 100
        return x.view(-1, hidden_size * 3)


class _MaxPooling(nn.Module):
    def __init__(self):
        super(_MaxPooling, self).__init__()

    def forward(self, x, hidden_size):
        x, _ = torch.max(x, dim=2)
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


# in pyTorch lstm takes a 3-dim tensor as input, where the first dim represents the sentence_len,
# the second dim represents the batch_size,, and the third dim represents the word embedding_size
class BLSTM(nn.Module):
    def __init__(self, config):
        super(BLSTM, self).__init__()
        self.config = config
        self.embedding = Embedding(config)
        # self.batch_size = config.batch_size  # bag size
        self.sentence_len = config.sentence_len  # 120
        self.dropout = config.dropout  # 0.1
        self.in_channels = config.in_channels  #
        self.out_channels = config.d_model // 2  # hidden dim // 2 = 512 // 2 = 256
        self.rnn = nn.LSTM(input_size=self.in_channels, hidden_size=self.out_channels, bidirectional=True)
        # self.hidden = self.init_hidden()  # (2, 120, 256)
        self.attn = TransformerEncoder(config)

    def init_hidden(self):
        if self.config.use_gpu:
            return torch.stack([torch.zeros(2, self.sentence_len, self.out_channels).cuda(),
                                torch.zeros(2, self.sentence_len, self.out_channels).cuda()])
        else:
            return torch.stack([torch.zeros(2, self.sentence_len, self.out_channels),
                                torch.zeros(2, self.sentence_len, self.out_channels)])

    def forward(self, embedding):
        """
        :param embedding: word embeddings sentences, (n, sen_len=120, embedding_size=60)
        """
        embedding = embedding.permute(1, 0, 2)  # (sen_len=120, n, embedding_size=60)
        lstm_out, self.hidden = self.rnn(embedding)  # lstm stores the final state for all
        x = lstm_out.permute(1, 0, 2)  # (120, n, 256) -> (n, sen_len=120, 256)
        x = self.attn(x)  # (n, sen_len=120, 256) ->  (n, sen_len=120, 256)
        x = torch.sum(x, 1).unsqueeze(0)  # (n, sen_len=120, 256) ->  (n, sen_len=120, 1)
        return x



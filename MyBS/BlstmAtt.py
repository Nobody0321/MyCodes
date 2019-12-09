import torch
import torch.nn as nn
from model import Embedding, Model
from Attention import TransformerEncoder, Bag_Attention


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


class BLstmAtt(Model):
    def __init__(self, config):
        super(BLstmAtt, self).__init__(config)
        self.encoder = BLSTM(config)
        self.selector = Bag_Attention(config)

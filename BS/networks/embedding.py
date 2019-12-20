import torch
import torch.nn as nn


# embed words to low dim vector by looking up a pre trained w2v matrix
class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding(self.config.data_word_vec.shape[0], self.config.data_word_vec.shape[1])
        if self.config.use_attn:
            self.pos0_embedding = nn.Embedding(self.config.max_sen_length, self.config.pos_embedding_dim, padding_idx=0)
        self.pos1_embedding = nn.Embedding(self.config.relative_pos_num, self.config.pos_embedding_dim, padding_idx=0)
        self.pos2_embedding = nn.Embedding(self.config.relative_pos_num, self.config.pos_embedding_dim, padding_idx=0)
        self.init_word_weights()
        self.init_pos_weights()
        self.word = None
        self.pos0 = None
        self.pos1 = None
        self.pos2 = None

    def init_word_weights(self):
        # load pre trained w2v matrix
        self.word_embedding.weight.data.copy_(torch.from_numpy(self.config.data_word_vec))

    def init_pos_weights(self):
        if self.config.use_attn:
            nn.init.xavier_uniform_(self.pos0_embedding.weight.data)
        nn.init.xavier_uniform_(self.pos1_embedding.weight.data)
        if self.pos1_embedding.padding_idx is not None:
            self.pos1_embedding.weight.data[self.pos1_embedding.padding_idx].fill_(0)
        nn.init.xavier_uniform_(self.pos2_embedding.weight.data)
        if self.pos2_embedding.padding_idx is not None:
            self.pos2_embedding.weight.data[self.pos2_embedding.padding_idx].fill_(0)

    def forward(self):
        word = self.word_embedding(self.word)
        pos1 = self.pos1_embedding(self.pos1)
        pos2 = self.pos2_embedding(self.pos2)
        if self.config.use_attn:
            pos0 = self.pos0_embedding(self.pos0)
            embedding = torch.cat((word, pos0, pos1, pos2), dim=2)
        else:
            embedding = torch.cat((word, pos1, pos2), dim=2)
        return embedding

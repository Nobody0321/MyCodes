import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from networks.encoder import TransformerEncoder


# embed words to low dim vector by looking up a pre trained w2v matrix
class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding(self.config.data_word_vec.shape[0], self.config.data_word_vec.shape[1])
        self.pos1_embedding = nn.Embedding(self.config.pos_num, self.config.pos_embedding_dim, padding_idx=0)
        self.pos2_embedding = nn.Embedding(self.config.pos_num, self.config.pos_embedding_dim, padding_idx=0)
        self.init_word_weights()
        self.init_pos_weights()
        self.word = None
        self.pos1 = None
        self.pos2 = None

    def init_word_weights(self):
        # load pre trained w2v matrix
        self.word_embedding.weight.data.copy_(torch.from_numpy(self.config.data_word_vec))

    def init_pos_weights(self):
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
        embedding = torch.cat((word, pos1, pos2), dim=2)
        return embedding


class BiGru(nn.Module):
    def __init__(self, config):
        super(BiGru, self).__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.sentence_len = config.sentence_len  # 120
        self.dropout = config.dropout  # 0.1
        self.input_dim = config.input_dim  #
        self.out_channels = config.d_model // 2  # hidden dim // 2 = 256 // 2 = 256
        self.rnn = nn.GRU(input_size=self.input_dim, hidden_size=self.out_channels, bidirectional=True)
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
        :param embedding: word embeddings sentences, (n=batch_size/sen_num, sen_len=120, embedding_size=60)
        """
        embedding = embedding.permute(1, 0, 2)  # (sen_len=120, n, embedding_size=60)
        lstm_final_all, final_hidden = self.rnn(embedding)  # lstm_out(120, n, 256) stores the final state for all
        del lstm_final_all
        final_hidden = final_hidden.view(1, -1, 256)  # (2, sen_num, 128) -> (1, sen_num, 256)
        # final_hidden = final_hidden.permute(1, 0, 2)  # (1, n, 256) -> (n, 1, 256)
        return final_hidden


class BagAttention(nn.Module):
    def __init__(self, config, encoder_output_dim):
        super(BagAttention, self).__init__(config, encoder_output_dim)
        self.attn = TransformerEncoder(config)
        self.config = config
        self.relation_matrix = nn.Embedding(self.config.num_classes, encoder_output_dim)
        self.bias = nn.Parameter(torch.Tensor(self.config.num_classes))
        # randomly initialize attention vector for soft attention, and optimize it during training
        self.attention_matrix = nn.Embedding(self.config.num_classes, encoder_output_dim)
        self.init_weights()
        self.scope = None
        self.attention_query = None
        self.label = None
        self.dropout = nn.Dropout(self.config.dropout)

    def init_weights(self):
        nn.init.xavier_uniform_(self.relation_matrix.weight.data)
        nn.init.normal_(self.bias)
        nn.init.xavier_uniform_(self.attention_matrix.weight.data)

    def get_logits(self, x):
        # pass encoder vector to a linear, to get classifier vec
        logits = torch.matmul(x, torch.transpose(self.relation_matrix.weight, 0, 1), ) + self.bias
        return logits

    def forward(self, x):
        """
        :param x: output of encoder, max pooled attention vectors of sentences in a bag, (1, 36, 512)
        :return:
        """
        all_bag_vec = []
        for i in range(len(self.scope) - 1):
            temp_x = x[:, self.scope[i]: self.scope[i + 1], :]  # (1, bag_size, 512)
            temp_x = self.attn(temp_x)  # (1, bag_size, 512) -> (1, bag_size, 512)
            temp_x = temp_x.max(dim=1)[0]  # (1, bag_size, 512) -> (1, 512)
            temp_x = temp_x.squeeze()  # (1, 512)/(1, 1, 512) -> (512)
            all_bag_vec.append(temp_x)
        all_bag_vec = torch.stack(all_bag_vec)  # (bags in batch, 512)
        all_bag_vec = self.dropout(all_bag_vec)
        logits = self.get_logits(all_bag_vec)  # (bags in batch, 512) -> (bags in batch, 52/relation_num)
        logits = F.softmax(logits, dim=1)  # (bags in batch, 53)
        return logits

    def test(self, x):
        all_bag_vec = []
        for i in range(len(self.scope) - 1):
            temp_x = x[:, self.scope[i]: self.scope[i + 1], :]  # (1, bag_size, 512)
            if temp_x.size(1) > 1:
                # this bag contains multi sentences/instances, we need to perform bag level attention and max pooling
                temp_x = self.attn(temp_x)  # (1, bag_size, 512) -> (1, bag_size, 512)
                temp_x = temp_x.max(dim=1)[0]  # (1, bag_size, 512) -> (1, 512)
            temp_x = temp_x.squeeze()  # (1, 512)/(1, 1, 512) -> (512)
            all_bag_vec.append(temp_x)
        all_bag_vec = torch.stack(all_bag_vec)  # (bags in batch, 512)
        # all_bag_vec = self.dropout(all_bag_vec)
        logits = self.get_logits(all_bag_vec)  # (bags in batch, 512) -> (bags in batch, 52/relation_num)
        logits = F.softmax(logits, dim=1)  # (bags in batch, 53)
        return list(logits.data.cpu().numpy())


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.config = config
        self.label = None
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits):
        """

        :param logits: selector output, (bag_num, 53/relation_num)
        :return: loss and predicted relation
        """
        loss = self.loss(logits, self.label)
        prediction = torch.max(logits, dim=1)[1]  # output: max indices, aka max relation
        return loss, prediction


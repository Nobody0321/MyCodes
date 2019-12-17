import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import TransformerEncoder


class Selector(nn.Module):
    def __init__(self, config, encoder_output_dim):
        """
        :param config: config object
        :param encoder_output_dim: output dim for encoder
        """
        super(Selector, self).__init__()
        self.config = config
        # the linear layer between selector and output
        self.relation_matrix = nn.Embedding(self.config.num_classes, encoder_output_dim)
        self.bias = nn.Parameter(torch.Tensor(self.config.num_classes))
        # randomly initialize attention vector for soft attention, and optimize it during training
        self.attention_matrix = nn.Embedding(self.config.num_classes, encoder_output_dim)
        self.init_weights()
        self.scope = None
        self.attention_query = None  # will be replaced with attention id in training
        self.label = None
        self.dropout = nn.Dropout(self.config.dropout)

    def init_weights(self):
        nn.init.xavier_uniform_(self.relation_matrix.weight.data)
        nn.init.normal_(self.bias)
        nn.init.xavier_uniform_(self.attention_matrix.weight.data)

    def get_logits(self, x):
        """
        pass encoder vector to a linear, to get classifier vec
        :param x:
        :return:
        """
        logits = torch.matmul(x, torch.transpose(self.relation_matrix.weight, 0, 1), ) + self.bias
        return logits

    def forward(self, x):
        raise NotImplementedError

    def test(self, x):
        raise NotImplementedError


class Attention(Selector):
    def _attention_train_logit(self, x):
        """

        :param x: sentences (1, bag_size, 256)
        :return: attention sum
        """
        # use relation ids to look up corresponding relation query vector (randomly initialized)
        relation_vector = self.relation_matrix(self.attention_query)
        # relation weight
        attention_wight = self.attention_matrix(self.attention_query)
        attention_logit = torch.sum(x * attention_wight * relation_vector, 1, True)
        return attention_logit

    def _attention_test_logit(self, x):
        """
        calculate attention vector for each sentence in bag
        :param x: sen vectors in a bag
        :return: attention sum vector for each sentence
        """
        attention_logit = torch.matmul(x, torch.transpose(self.attention_matrix.weight * self.relation_matrix.weight, 0,
                                                          1))
        return attention_logit

    def forward(self, x):
        attention_logit = self._attention_train_logit(x)
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i]: self.scope[i + 1]], 0, 1), 1)
            final_bag_feature = torch.squeeze(torch.matmul(attention_score, sen_matrix))
            tower_repre.append(final_bag_feature)
        stack_repre = torch.stack(tower_repre)
        stack_repre = self.dropout(stack_repre)
        logits = self.get_logits(stack_repre)
        return logits

    def test(self, x):
        attention_logit = self._attention_test_logit(x)
        tower_output = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i]: self.scope[i + 1]], 0, 1), 1)
            final_repre = torch.matmul(attention_score, sen_matrix)
            logits = self.get_logits(final_repre)
            tower_output.append(torch.diag(F.softmax(logits, 1)))
        stack_output = torch.stack(tower_output)
        return list(stack_output.data.cpu().numpy())


class One(Selector):
    def forward(self, x):
        tower_logits = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            sen_matrix = self.dropout(sen_matrix)
            logits = self.get_logits(sen_matrix)
            score = F.softmax(logits, 1)
            _, k = torch.max(score, dim=0)
            k = k[self.label[i]]
            tower_logits.append(logits[k])
        return torch.cat(tower_logits, 0)

    def test(self, x):
        tower_score = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            logits = self.get_logits(sen_matrix)
            score = F.softmax(logits, 1)
            score, _ = torch.max(score, 0)
            tower_score.append(score)
        tower_score = torch.stack(tower_score)
        return list(tower_score.data.cpu().numpy())


class Average(Selector):
    def forward(self, x):
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            final_repre = torch.mean(sen_matrix, 0)
            tower_repre.append(final_repre)
        stack_repre = torch.stack(tower_repre)
        stack_repre = self.dropout(stack_repre)
        logits = self.get_logits(stack_repre)
        return logits

    def test(self, x):
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            final_repre = torch.mean(sen_matrix, 0)
            tower_repre.append(final_repre)
        stack_repre = torch.stack(tower_repre)
        logits = self.get_logits(stack_repre)
        score = F.softmax(logits, 1)
        return list(score.data.cpu().numpy())


class BagAttention(Selector):
    def __init__(self, config, encoder_output_dim):
        super(BagAttention, self).__init__(config, encoder_output_dim)
        self.attn = TransformerEncoder(config=config)
        self.BNorm = nn.BatchNorm1d(num_features=encoder_output_dim)

    def forward(self, x):
        """
        :param x: output of encoder, max pooled attention vectors of sentences in a bag, (1, 36, 256)
        :return:
        """
        all_bag_vec = []
        for i in range(len(self.scope) - 1):
            temp_x = x[:, self.scope[i]: self.scope[i + 1], :]  # (1, bag_size, 256)
            # this bag contains multi sentences/instances, we need to perform bag level attention and max pooling
            temp_x = self.attn(temp_x)  # (1, bag_size, 256) -> (1, bag_size, 256)
            temp_x = temp_x.max(dim=1)[0]  # (1, bag_size, 256) -> (1, 256)
            # temp_x = self.BNorm(temp_x)
            temp_x = temp_x.squeeze()  # (1, 256)/(1, 1, 256) -> (256)
            all_bag_vec.append(temp_x)
        all_bag_vec = torch.stack(all_bag_vec)  # (bags in batch, 256)
        all_bag_vec = self.dropout(all_bag_vec)
        logits = self.get_logits(all_bag_vec)  # (bags in batch, 256) -> (bags in batch, 52/relation_num)
        del all_bag_vec
        # logits = F.softmax(logits, dim=1)  # (bags in batch, 53)
        return logits

    def test(self, x):
        all_bag_vec = []
        for i in range(len(self.scope) - 1):
            temp_x = x[:, self.scope[i]: self.scope[i + 1], :]  # (1, bag_size, 256)
            if temp_x.size(1) > 1:
                # this bag contains multi sentences/instances, we need to perform bag level attention and max pooling
                temp_x = self.attn(temp_x)  # (1, bag_size, 256) -> (1, bag_size, 256)
                temp_x = temp_x.max(dim=1)[0]  # (1, bag_size, 256) -> (1, 256)
            temp_x = temp_x.squeeze()  # (1, 256)/(1, 1, 256) -> (256)
            all_bag_vec.append(temp_x)
        all_bag_vec = torch.stack(all_bag_vec)  # (bags in batch, 256)
        # all_bag_vec = self.dropout(all_bag_vec)
        logits = self.get_logits(all_bag_vec)  # (bags in batch, 256) -> (bags in batch, 52/relation_num)
        # del all_bag_vec
        logits = F.softmax(logits, dim=1)  # (bags in batch, 53)
        return list(logits.data.cpu().numpy())


class LayerAtt(nn.Module):
    def __init__(self, config, encoder_output_dim):
        super(LayerAtt, self).__init__()
        self.sen_bag_attn = Attention(config, encoder_output_dim)
        self.super_bag_attn = Attention(config, encoder_output_dim)

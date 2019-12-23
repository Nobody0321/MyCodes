import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import SelfAttention, SelfMaxAttEncoder


class Selector(nn.Module):
    def __init__(self, config, encoder_output_dim):
        """
        :param config: config class
        :param encoder_output_dim:
        """
        super(Selector, self).__init__()
        self.config = config
        # the linear layer between selector and output
        self.relation_matrix = nn.Embedding(self.config.num_classes, encoder_output_dim)
        self.bias = nn.Parameter(torch.Tensor(self.config.num_classes))
        # attention scores for each dim in encoder_output(230) for each relation
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
        :param x: (batch_size, 230)
        :return:
        """
        # (batch_size, 230) * (230, 53) -> (batch_size, 53)
        logits = torch.matmul(x, torch.transpose(self.relation_matrix.weight, 0, 1), ) + self.bias
        return logits

    def forward(self, x):
        raise NotImplementedError

    def test(self, x):
        raise NotImplementedError


class Attention(Selector):
    def _attention_train_logit(self, x):
        relation_query = self.relation_matrix(self.attention_query)
        attention = self.attention_matrix(self.attention_query)
        attention_logit = torch.sum(x * attention * relation_query, 1, True)
        return attention_logit

    def _attention_test_logit(self, x):
        attention_logit = torch.matmul(x, torch.transpose(self.attention_matrix.weight * self.relation_matrix.weight, 0,
                                                          1))
        return attention_logit

    def forward(self, x):
        attention_logit = self._attention_train_logit(x)
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i]: self.scope[i + 1]], 0, 1), 1)
            final_repre = torch.squeeze(torch.matmul(attention_score, sen_matrix))
            tower_repre.append(final_repre)
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


class SelfAttMaxSelector(nn.Module):
    def __init__(self, config, input_dim):
        super(SelfAttMaxSelector, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.dropout = nn.Dropout(self.config.dropout)
        self.attn = SelfMaxAttEncoder(config=config, input_dim=self.input_dim, output_dim=self.output_dim)
        self.linear = nn.Linear(self.output_dim, self.config.num_classes)

    def forward(self, x):
        """

        :param x: output of sentence encoder, (n, 230)
        :return:
        """
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_vec_one_bag = x[self.scope[i]: self.scope[i + 1]]  # (bag_size, 230)
            sen_vec_one_bag = sen_vec_one_bag.unsqueeze(0)  # (1, bag_size, 230)
            final_repre = self.attn(sen_vec_one_bag).squeeze()  # (1, bag_size, 230) -> (1, 230) -> (230)
            tower_repre.append(final_repre)
        stack_repre = torch.stack(tower_repre)  # (batch_size, 230)
        stack_repre = self.dropout(stack_repre)
        logits = self.linear(stack_repre)  # (batch_size, 230) * (230, 53) -> (batch_size, 53)
        return logits

    def test(self, x):
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_vec_one_bag = x[self.scope[i]: self.scope[i + 1]]  # (bag_size, 230)
            sen_vec_one_bag = sen_vec_one_bag.unsqueeze(0)  # (1, bag_size, 230)
            final_repre = self.attn(sen_vec_one_bag).squeeze()  # (230)
            tower_repre.append(final_repre)
        stack_repre = torch.stack(tower_repre)  # batch_size 230
        logits = self.linear(stack_repre)
        score = F.softmax(logits, 1)
        return list(score.data.cpu().numpy())


class SelfSoftAttSelector(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super(SelfSoftAttSelector, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(self.config.dropout)
        self.self_attn = SelfAttention(config=config, input_dim=self.input_dim, output_dim=self.output_dim)
        self.soft_attn = Attention(self.config, self.config.hidden_dim)
        self.scope = None
        self.attention_query = None  # will be replaced with attention id in training
        self.label = None

    def forward(self, x):
        """

        :param x: output of sentence encoder, (n, 230)
        :return:
        """
        # tower_repre = []
        self.soft_attn.scope = self.scope
        self.soft_attn.attention_query = self.attention_query  # will be replaced with attention id in training
        self.soft_attn.label = self.label
        stack_repre = None
        for i in range(len(self.scope) - 1):
            sen_vec_one_bag = x[self.scope[i]: self.scope[i + 1]]  # (bag_size, 230)
            final_repre = self.self_attn(
                sen_vec_one_bag.unsqueeze(0)).squeeze(0)  # (1, bag_size, 230) -> (bag_size, 230)
            if stack_repre is None:
                stack_repre = final_repre
            else:
                stack_repre = torch.cat((stack_repre, final_repre), dim=0)
        stack_repre = self.dropout(stack_repre)
        logits = self.soft_attn(stack_repre)  # (n, 53) -> (batch_size, 53)
        return logits

    def test(self, x):
        self.soft_attn.scope = self.scope
        stack_repre = None
        for i in range(len(self.scope) - 1):
            sen_vec_one_bag = x[self.scope[i]: self.scope[i + 1]]  # (bag_size, 230)
            final_repre = self.self_attn(
                sen_vec_one_bag.unsqueeze(0)).squeeze(0)  # (1, bag_size, 230) -> (bag_size, 230)
            if stack_repre is None:
                stack_repre = final_repre
            else:
                stack_repre = torch.cat((stack_repre, final_repre), dim=0)
        stack_repre = self.dropout(stack_repre)
        scores = self.soft_attn.test(stack_repre)
        return scores

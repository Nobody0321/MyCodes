import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import SelfAttEncoder


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


class SoftAttention(Selector):
    def _attention_train_logit(self, bag_vec):
        """
        calculate attention logit
        :param bag_vec: sentences (n, 230)
        :return: attention sum
        """
        # use relation ids to look up corresponding relation query vector (randomly initialized)
        relation_vector = self.relation_matrix(self.attention_query)  # (n, 230)  # each sen looks up its relation vec
        # relation weight
        attention_wight = self.attention_matrix(self.attention_query)  # (n, 230)
        attention_logit = torch.sum(bag_vec * attention_wight * relation_vector, 1, True)  # (n, 1)
        return attention_logit

    def _attention_test_logit(self, bag_vec):
        """
        calculate attention logit each sentence in bag
        :param bag_vec: sen vectors in a bag : (n, 230)
        :return: attention sum vector for each sentence
        """
        # (n, 230) * (230, 53) -> (n, 53)
        # do not perform sum here
        attention_logit = torch.matmul(bag_vec,
                                       torch.transpose(self.attention_matrix.weight * self.relation_matrix.weight, 0,
                                                       1))
        return attention_logit

    def forward(self, x):
        """
        perform soft attention
        :param x: (n=sen_num in bag, 230)
        :return:
        """
        attention_logit = self._attention_train_logit(x)  # attention score (batch_size, 1)
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]  # (bag_size, 230)
            attention_weight = F.softmax(torch.transpose(attention_logit[self.scope[i]: self.scope[i + 1]], 0, 1),
                                         1)  # (1, bag_size)
            final_bag_feature = torch.squeeze(torch.matmul(attention_weight,
                                                           sen_matrix))  # (1, 230) -> (230), do not perform sum here, leave it to classifier
            tower_repre.append(final_bag_feature)
        stack_repre = torch.stack(tower_repre)  # (batch_size, 230)
        stack_repre = self.dropout(stack_repre)  # (batch_size, 230)
        logits = self.get_logits(stack_repre)  # (batch_size, 53)
        return logits

    def test(self, x):
        """
        perform soft attention for testing
        :param x: (n, 230)
        :return:
        """
        attention_logit = self._attention_test_logit(x)  # (n, 53)  # 一个batch中所有句子对于所有关系的attention score
        tower_output = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]  # (bag_size, 230)

            # 一个relation对于bag中所有句子的attention weight
            attention_weight = F.softmax(torch.transpose(attention_logit[self.scope[i]: self.scope[i + 1]], 0, 1),
                                         1)  # (bag_size, 53) ->  (53, bag_size)
            # final repre 的第i行表示对应关系i的经过soft attention的bag vector  每一个关系对应的bag vector
            final_repre = torch.matmul(attention_weight,
                                       sen_matrix)  # (53, bag_size) * (bag_size, 230) -> (53, 230)  # use attention score to integrate bag
            # 每一个关系对应的bag vector，对应的所有关系的得分
            logits = self.get_logits(final_repre)  # (53, 53)
            # 取对角线得到每一个relation对应的bag vector 对应的 这个relation的得分，就是这个bag在所有53个relation的得分
            tower_output.append(torch.diag(F.softmax(logits, 1)))  # (53)  # 对角线上的元素表示了每一个关系对应的attention score?
        stack_output = torch.stack(tower_output)  # (batch_num, 53)
        return list(stack_output.data.cpu().numpy())


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


class SelfAttSelector(Selector):
    def __init__(self, config, input_dim, output_dim):
        super(SelfAttSelector, self).__init__(config, output_dim)
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.attn = SelfAttEncoder(config=config, input_dim=self.input_dim, output_dim=self.output_dim)

    def forward(self, x):
        """

        :param x: output of sentence encoder, (n, 230)
        :return:
        """
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_vec_one_bag = x[self.scope[i]: self.scope[i + 1]]  # (bag_size, 230)
            sen_vec_one_bag = sen_vec_one_bag.unsqueeze(0)  # (1, bag_size, 230)
            final_repre = self.attn(sen_vec_one_bag).squeeze()  # (1, bag_size, 230) -> (230)
            tower_repre.append(final_repre)
        stack_repre = torch.stack(tower_repre)
        stack_repre = self.dropout(stack_repre)
        logits = self.get_logits(stack_repre)
        return logits

    def test(self, x):
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_vec_one_bag = x[self.scope[i]: self.scope[i + 1]]  # (bag_size, 230)
            sen_vec_one_bag = sen_vec_one_bag.unsqueeze(0)  # (1, bag_size, 230)
            final_repre = self.attn(sen_vec_one_bag).squeeze()  # (230)
            tower_repre.append(final_repre)
        stack_repre = torch.stack(tower_repre)  # batch_size 230
        logits = self.get_logits(stack_repre)
        score = F.softmax(logits, 1)
        return list(score.data.cpu().numpy())

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Selector(nn.Module):
    def __init__(self, config, relation_dim):
        super(Selector, self).__init__()
        self.config = config
        self.relation_matrix = nn.Embedding(self.config.num_classes, relation_dim)  # 53, d
        self.bias = nn.Parameter(torch.Tensor(self.config.num_classes))
        self.attention_matrix = nn.Embedding(self.config.num_classes, relation_dim)
        self.init_weights()
        self.scope = None
        self.attention_query = None
        self.label = None
        self.dropout = nn.Dropout(self.config.drop_prob)

    def init_weights(self):
        nn.init.xavier_uniform_(self.relation_matrix.weight.data)
        nn.init.normal_(self.bias)
        nn.init.xavier_uniform_(self.attention_matrix.weight.data)

    def get_logits(self, x):
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
        attention_logit = self._attention_train_logit(x)  # n, 53
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
        attention_logit = self._attention_test_logit(x)  # n, 53
        tower_output = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]  # b, d
            #  53, b
            attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i]: self.scope[i + 1]], 0, 1), 1)
            final_repre = torch.matmul(attention_score, sen_matrix)  # 53, d
            logits = self.get_logits(final_repre)  # 53, 53
            tower_output.append(torch.diag(F.softmax(logits, 1)))  # 53
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


class SenSoftAtt(nn.Module):
    """
    use self soft attention to calculate attention weights
    """

    def __init__(self, config):
        super(SenSoftAtt, self).__init__()
        self.config = config
        self.wei_mat = nn.Parameter(torch.randn(config.encoder_output_dim, 1))
        torch.nn.init.xavier_normal_(self.wei_mat)

    def forward(self, x):
        """

        :param x: n, 120, 230
        :return:
        """
        W = torch.tanh(x)
        alpha = F.softmax(torch.matmul(W.view(-1, self.config.encoder_output_dim), self.wei_mat), dim=1).view(-1, 1, self.config.max_length)  # n, 1, 120
        x = torch.bmm(alpha, x)  # n, 1, 230
        return x.squeeze(1)  # n, 230


class SenLevelAtt(Selector):
    """
    use word-relation alignment to calculate attention weights
    """

    def __init__(self, config):
        super(SenLevelAtt, self).__init__(config, config.encoder_output_dim)
        self.config = config
        self.assemble_matrix = nn.Parameter(torch.randn(config.num_classes, config.encoder_output_dim, config.encoder_output_dim))  # 53, 230*230
        torch.nn.init.xavier_normal_(self.assemble_matrix.data)
        self.activation = torch.tanh
        self.bn = nn.BatchNorm1d()

    def forward(self, x):
        """
        :param x: n, 120, 230 self att output
        :return:
        """
        # do sentence level att
        # n, 230, 1
        relation_query = self.relation_matrix(self.attention_query).unsqueeze(-1)
        # n, 230, 230
        assemble = self.assemble_matrix[self.attention_query]
        # n, 120, 1
        att_weights = torch.bmm(torch.bmm(x, assemble), relation_query)
        att_weights = self.dropout(att_weights)
        att_weights = F.softmax(att_weights, dim=1).permute(0, 2, 1)  # n, 120, 1 -> n, 1, 120
        x = torch.bmm(att_weights, x).squeeze(1)  # n, 1, 230 -> n, 230
        # do bag level att
        x = self.activation(x)
        x = self.dropout(x)
        sentence_bag_attention = self.attention_matrix(self.attention_query)
        attention_logit = torch.sum(x * sentence_bag_attention * relation_query.squeeze(-1), 1, True)
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
        """
        in test, we need to store all probability, for pr curve
        :param x: n, l, 230 self att output
        :return:
        """
        batch_num = x.size(0)
        sen_len = x.size(1)
        x = x.view(-1, x.size(-1))  # n*L, 230
        # calculate W*r first,  # 53, 230, 23 * 53,230,1 -> 53, 230
        ar = torch.bmm(self.assemble_matrix.view(self.config.num_classes, self.config.encoder_output_dim, -1)
                       , self.relation_matrix.weight.data.unsqueeze(-1)).squeeze(-1)
        # n*l, 53
        sen_attention_logit = torch.matmul(x, torch.transpose(ar, 0, 1))
        # n*l, 53->n, l, 53
        # n个句子，每个句子针对53个关系有53种加权和方式
        sen_attention_weights = F.softmax(sen_attention_logit, dim=1).view(batch_num, sen_len, -1)
        # n, (53, l) * (l, 230) = n, 53, 230  n个句子，每个句子的53种加权和结果
        x = torch.bmm(sen_attention_weights.permute(0, 2, 1), x.view(batch_num, sen_len, -1))
        x = self.activation(x)
        filtered_sen = []
        for all_sum_sen_vec in x:
            # all_sum_sen_vec : 53, 230
            logits = self.get_logits(all_sum_sen_vec)  # 53, 53
            idx = torch.argmax(torch.diag(F.softmax(logits, 1)))  # 获取这个句子最有可能的关系
            filtered_sen.append(all_sum_sen_vec[idx])
        # n, 230
        x = torch.stack(filtered_sen)
        x = self.activation(x)
        attention_logit = torch.matmul(x, torch.transpose(self.attention_matrix.weight * self.relation_matrix.weight, 0, 1))
        tower_output = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]  # b, d
            #  53, b
            attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i]: self.scope[i + 1]], 0, 1), 1)
            final_repre = torch.matmul(attention_score, sen_matrix)  # 53, d
            logits = self.get_logits(final_repre)  # 53, 53
            tower_output.append(torch.diag(F.softmax(logits, 1)))  # 53
        stack_output = torch.stack(tower_output)
        return list(stack_output.data.cpu().numpy())


class SenSoftAndBagSoftAttention(nn.Module):
    def __init__(self, config):
        super(SenSoftAndBagSoftAttention, self).__init__()
        self.sen_attn = SenSoftAtt(config)
        self.bag_attn = Attention(config, config.encoder_output_dim)
        self.scope = None
        self.attention_query = None
        self.label = None

    def forward(self, x):
        """

        :param x: n, 120, 230
        :return:
        """
        self.bag_attn.scope = self.scope
        self.bag_attn.attention_query = self.attention_query
        self.bag_attn.label = self.label
        x = self.sen_attn(x)  # n, 120, 230 -> n, 230
        x = self.bag_attn(x)
        return x

    def test(self, x):
        """

        :param x:  n, 120, 230
        :return:
        """
        self.bag_attn.scope = self.scope
        self.bag_attn.label = self.label
        x = self.sen_attn(x)  # n, 120, 230 -> n, 230
        scores = self.bag_attn.test(x)
        return scores


from networks.attention import AttEncoder


class SelfAttSelector(Selector):
    def __init__(self, config, relation_dim):
        super(SelfAttSelector, self).__init__(config, relation_dim)
        self.bag_attn = AttEncoder(d_model=relation_dim, n_heads=config.n_attn_heads, d_output=relation_dim,
                                        dropout=config.attn_dropout)

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
            sen_matrix = self.bag_attn(sen_matrix.unsqueeze(0)).squeeze(0)
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
            sen_matrix = self.bag_attn(sen_matrix.unsqueeze(0)).squeeze(0)
            attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i]: self.scope[i + 1]], 0, 1), 1)
            final_repre = torch.matmul(attention_score, sen_matrix)
            logits = self.get_logits(final_repre)
            tower_output.append(torch.diag(F.softmax(logits, 1)))
        stack_output = torch.stack(tower_output)
        return list(stack_output.data.cpu().numpy())


class SelfAttMaxSelector(Selector):
    def __init__(self, config, relation_dim):
        super(SelfAttMaxSelector, self).__init__(config, relation_dim)
        self.bag_attn = AttEncoder(d_model=relation_dim, n_heads=config.n_attn_heads, d_output=relation_dim,
                                        dropout=config.attn_dropout)

    def forward(self, x):
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            sen_matrix = self.bag_attn(sen_matrix.unsqueeze(0)).squeeze(0).max(0)[0]
            tower_repre.append(sen_matrix)
        stack_repre = torch.stack(tower_repre)
        stack_repre = self.dropout(stack_repre)
        logits = self.get_logits(stack_repre)
        return logits

    def test(self, x):
        tower_output = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            sen_matrix = self.bag_attn(sen_matrix.unsqueeze(0)).squeeze(0).max(0)[0]
            logits = self.get_logits(sen_matrix)
            tower_output.append(F.softmax(logits, 1))
        stack_output = torch.stack(tower_output)
        return list(stack_output.data.cpu().numpy())

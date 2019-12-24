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
		self.relation_matrix = nn.Embedding(self.config.num_classes, relation_dim)
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


class SenSoftAtt(nn.Module):
	def __init__(self, config):
		super(SenSoftAtt, self).__init__()
		self.config = config
		self.assemble_matrix = nn.Parameter(torch.Tensor(1, config.encoder_output_dim))  # 230, 230to calculate
		self.init_weights()
		self.scope = None
		self.attention_query = None  # sentence one hot label
		self.label = None

	def init_weights(self):
		nn.init.xavier_uniform_(self.assemble_matrix.data)

	def forward(self, x):
		"""n, 120, 230"""
		M = torch.tanh(x).view(230, -1)  # 230 n*120
		weights = F.softmax(torch.matmul(self.assemble_matrix, M), dim=0).view(-1, 1, 120)  # n 1 120
		x = torch.bmm(weights, x).squeeze(1) # n, 1, 230
		return torch.tanh(x)


class SenSoftAndBagSoftAttention(nn.Module):
	def __init__(self, config):
		super(SenSoftAndBagSoftAttention, self).__init__()
		self.sen_attn = SenSoftAtt(config)
		self.bag_attn = Attention(config, config.encoder_output_dim)
		self.scope = None
		self.attention_query = None
		self.label = None

	def init_weights(self):
		nn.init.xavier_uniform_(self.relation_matrix.weight.data)
		nn.init.normal_(self.bias)
		nn.init.xavier_uniform_(self.attention_matrix.weight.data)

	def forward(self, x):
		"""

		:param x: n, 120, 230
		:return:
		"""
		self.sen_attn.scope = self.scope
		self.sen_attn.attention_query = self.attention_query
		self.sen_attn.label = self.label
		self.bag_attn.scope = self.scope
		self.bag_attn.attention_query = self.attention_query
		self.bag_attn.label = self.label
		x = self.sen_attn(x)  # n, 120, 230 -> n, 230
		return self.bag_attn(x)

	def test(self, x):
		"""

		:param x:  n, 120, 230
		:return:
		"""
		self.sen_attn.scope = self.scope
		self.sen_attn.attention_query = self.attention_query
		self.sen_attn.label = self.label
		self.bag_attn.scope = self.scope
		self.bag_attn.attention_query = self.attention_query
		self.bag_attn.label = self.label
		x = self.sen_attn(x)  # n, 120, 230 -> n, 230
		scores = self.bag_attn.test(x)
		return scores


from networks.attention import AttEncoderBlock


class SelfAttSelector(Selector):
	def __init__(self, config, relation_dim):
		super(SelfAttSelector, self).__init__(config, relation_dim)
		self.bag_attn = AttEncoderBlock(d_model=relation_dim, n_heads=config.n_attn_heads, d_output=relation_dim,
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
		self.bag_attn = AttEncoderBlock(d_model=relation_dim, n_heads=config.n_attn_heads, d_output=relation_dim,
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

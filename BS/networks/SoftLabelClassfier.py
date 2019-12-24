import torch
import torch.nn as nn


class SoftLabelClassifier(nn.Module):

	def __init__(self, config, weights=[]):
		super(SoftLabelClassifier, self).__init__()
		self.config = config
		self.weights = torch.tensor(weights)
		if self.weights.shape[0] == 0:
			self.weights = torch.tensor([0.9] + [0.7] * (config.num_classes - 1))
		self.loss = nn.CrossEntropyLoss()

	def forward(self, logits):
		soft_label = self.cal_soft_label(logits, self.label)
		loss = self.loss(logits, soft_label)
		_, output = torch.max(logits, dim=1)
		return loss, output.data

	def cal_soft_label(self, logits, gt_label):
		soft_label_id = torch.argmax(logits + max(logits) * self.weithgs * gt_label)
		soft_label = torch.zeros(self.config.num_classes)
		soft_label[soft_label_id] = 1
		return soft_label
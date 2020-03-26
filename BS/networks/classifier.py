import torch
import torch.nn as nn
from ..networks.focal_loss import FocalLoss


class Classifier(nn.Module):
	def __init__(self, config):
		super(Classifier, self).__init__()
		self.config = config
		self.label = None
		self.loss = nn.CrossEntropyLoss()

	def forward(self, logits):
		loss = self.loss(logits, self.label)
		_, output = torch.max(logits, dim=1)
		return loss, output.data


class ClassifierFocal(nn.Module):
	def __init__(self, config):
		super(ClassifierFocal, self).__init__()
		self.config = config
		self.label = None
		self.loss = FocalLoss(num_classes=config.num_classes)

	def forward(self, logits):
		loss = self.loss(logits, self.label)
		_, output = torch.max(logits, dim=1)
		return loss, output.data

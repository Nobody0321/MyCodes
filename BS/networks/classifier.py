import torch
import torch.nn as nn
from .focal_loss import FocalLoss


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.config = config
        self.label = None
        # self.loss = nn.CrossEntropyLoss()
        self.loss = FocalLoss(num_classes=config.num_classes)

    def forward(self, logits):
        """
        :param logits: selector output, before softmax (batch_size, 53/relation_num)
        :return: loss and predicted relation
        """
        loss = self.loss(logits, self.label)
        prediction = torch.max(logits, dim=1)[1]  # max indices, aka max relation
        return loss, prediction

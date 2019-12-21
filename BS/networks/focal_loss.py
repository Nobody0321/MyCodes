from torch import nn
import torch
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=53, size_average=True):
        """
        FocalLoss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        FocalLoss损失计算
        :param preds:   预测类别. size:(n,53)
        :param labels:  实际类别. size:(n)
        :return:
        """
        preds = preds.view(-1, preds.size(-1))  # if dim of preds == 3, concatenate in batch (n, 53)
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)  # calculate softmax (n, 53)
        preds_logsoft = torch.log(preds_softmax)  # calculate log of each logit (n, 53)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # select the score for gt label
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))  # select the loss for gt label
        # 以上计算原始交叉熵完毕
        alpha = self.alpha.gather(0, labels.view(-1))  # get alpha for gt label
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)  # (1-pt)**γ
        loss = torch.mul(alpha, loss.t())  # α * (1-pt)**γ
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

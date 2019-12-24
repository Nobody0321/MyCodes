from networks.encoder import SelfPCNN
from networks.selector import Attention
from networks.classifier import ClassifierFocal
from .Model import Model


class SelfPCNN_ATT(Model):
	def __init__(self, config):
		super(SelfPCNN_ATT, self).__init__(config)
		self.encoder = SelfPCNN(config)
		self.selector = Attention(config, config.hidden_size * 3)
		# self.classifier = ClassifierFocal(config)

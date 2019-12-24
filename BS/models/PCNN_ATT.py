from networks.encoder import PCNN
from networks.selector import Attention
from networks.classifier import ClassifierFocal
from .Model import Model


class PCNN_ATT(Model):
	def __init__(self, config):
		super(PCNN_ATT, self).__init__(config)
		self.encoder = PCNN(config)
		self.selector = Attention(config, config.hidden_size * 3)
		# self.classifier = ClassifierFocal(config)

from networks.encoder import *
from networks.selector import *
from .Model import Model


class PCNN_Self(Model):
	def __init__(self, config):
		super(PCNN_Self, self).__init__(config)
		self.encoder = PCNN(config)
		self.selector = SelfAttSelector(config, config.hidden_size * 3)

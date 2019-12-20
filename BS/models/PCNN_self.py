from networks.encoder import *
from networks.selector import *
from .Model import Model


class PCNN_self(Model):
	def __init__(self, config):
		super(PCNN_self, self).__init__(config)
		self.encoder = PCNN(config)
		self.selector = SelfAttSelector(config, config.hidden_size * 3, 230)

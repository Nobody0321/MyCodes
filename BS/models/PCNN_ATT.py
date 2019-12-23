from networks.encoder import *
from networks.selector import *
from .Model import Model


class PCNN_ATT(Model):
	def __init__(self, config):
		super(PCNN_ATT, self).__init__(config)
		self.encoder = PCNN(config)
		self.selector = Attention(config, config.hidden_dim * 3)

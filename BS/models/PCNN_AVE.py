from networks.encoder import PCNN
from networks.selector import Average
from .Model import Model


class PCNN_AVE(Model):
	def __init__(self, config):
		super(PCNN_AVE, self).__init__(config)
		self.encoder = PCNN(config)
		self.selector = Average(config, config.hidden_dim * 3)

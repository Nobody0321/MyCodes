from networks.encoder import PCNN
from networks.selector import One
from .Model import Model


class PCNN_ONE(Model):
	def __init__(self, config):
		super(PCNN_ONE, self).__init__(config)
		self.encoder = PCNN(config)
		self.selector = One(config, config.hidden_size * 3)

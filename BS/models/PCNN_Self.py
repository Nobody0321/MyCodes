from networks.encoder import PCNN
from networks.selector import SelfAttSelector
from .Model import Model


class PCNN_Self(Model):
	def __init__(self, config):
		super(PCNN_Self, self).__init__(config)
		self.encoder = PCNN(config)
		self.selector = SelfAttSelector(config, config.hidden_dim * 3)

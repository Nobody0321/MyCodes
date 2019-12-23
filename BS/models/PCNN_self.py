from networks.encoder import PCNN
from networks.selector import SelfSoftAttSelector
from .Model import Model


class PCNN_self(Model):
	def __init__(self, config):
		super(PCNN_self, self).__init__(config)
		self.encoder = PCNN(config)
		self.selector = SelfSoftAttSelector(config, config.hidden_dim * 3, config.hidden_dim)

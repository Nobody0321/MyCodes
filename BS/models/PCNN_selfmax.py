from networks.encoder import PCNN
from networks.selector import SelfAttMaxSelector
from .Model import Model


class PCNN_selfmax(Model):
	def __init__(self, config):
		super(PCNN_selfmax, self).__init__(config)
		self.encoder = PCNN(config)
		self.selector = SelfAttMaxSelector(config, input_dim=config.hidden_dim*3)

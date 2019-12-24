from networks.encoder import PCNN
from networks.selector import SelfAttMaxSelector
from .Model import Model


class PCNN_SelfMax(Model):
	def __init__(self, config):
		super(PCNN_SelfMax, self).__init__(config)
		self.encoder = PCNN(config)
		self.selector = SelfAttMaxSelector(config, input_dim=config.hidden_dim*3)

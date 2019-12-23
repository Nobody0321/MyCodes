from networks.encoder import CNN
from networks.selector import One
from .Model import Model


class CNN_ONE(Model):
	def __init__(self, config):
		super(CNN_ONE, self).__init__(config)
		self.encoder = CNN(config)
		self.selector = One(config, config.hidden_dim)

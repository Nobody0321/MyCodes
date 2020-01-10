from networks.encoder import *
from networks.selector import *
from .Model import Model


class CNN_AVE(Model):
	def __init__(self, config):
		super(CNN_AVE, self).__init__(config)
		self.encoder = CNN(config)
		self.selector = Average(config, config.hidden_size)

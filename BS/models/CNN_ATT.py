from networks.selector import SelfSelectiveAttention
from networks.encoder import *
from .Model import Model

class CNN_ATT(Model):
	def __init__(self, config):
		super(CNN_ATT, self).__init__(config)
		self.encoder = CNN(config)
		self.selector = SelfSelectiveAttention(config, config.hidden_size)

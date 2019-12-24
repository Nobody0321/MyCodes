from networks.encoder import *
from networks.selector import *
from .Model import Model


class MSNET_ATT(Model):
	def __init__(self, config):
		super(MSNET_ATT, self).__init__(config)
		self.encoder = SelfAttEncoderWithMax(config)
		self.selector = Attention(config, config.hidden_size)

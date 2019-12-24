from networks.encoder import *
from networks.selector import *
from .Model import Model


class MSNET_Layer_ATT(Model):
	def __init__(self, config):
		super(MSNET_Layer_ATT, self).__init__(config)
		self.encoder = SelfAttEncoder(config)
		self.selector = SenSoftAndBagSoftAttention(config)

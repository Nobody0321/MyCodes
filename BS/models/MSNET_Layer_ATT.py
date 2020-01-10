from networks.encoder import SelfAttEncoder
from networks.selector import SenSoftAndBagSoftAttention
from .Model import Model


class MSNET_Layer_ATT(Model):
	def __init__(self, config):
		super(MSNET_Layer_ATT, self).__init__(config)
		self.encoder = SelfAttEncoder(config)
		self.selector = SenSoftAndBagSoftAttention(config)

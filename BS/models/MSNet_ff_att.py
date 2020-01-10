from networks.encoder import SelfAttEncoder
from networks.selector import SenSoftAndBagSoftAttention
from .Model import Model


class Msnet_ff_att(Model):
	def __init__(self, config):
		super(Msnet_ff_att, self).__init__(config)
		self.encoder = SelfAttEncoder(config)
		self.selector = SenSoftAndBagSoftAttention(config)

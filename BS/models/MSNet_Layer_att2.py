from networks.encoder import SelfAttEncoder
from networks.selector import SenLevelAtt
from .Model import Model


class MSNet_Layer_att2(Model):
	def __init__(self, config):
		super(MSNet_Layer_att2, self).__init__(config)
		self.encoder = SelfAttEncoder(config)
		self.selector = SenLevelAtt(config)

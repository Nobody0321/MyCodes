from networks.encoder import BiGru_Att
from networks.selector import SenSoftAndBagSoftAttention
from .Model import Model


class BiGru_ATT(Model):
	def __init__(self, config):
		super(BiGru_ATT, self).__init__(config)
		self.encoder = BiGru_Att(config)
		self.selector = SenSoftAndBagSoftAttention(config)
		# self.selector = Attention(config, config.hidden_size)
		# self.classifier = ClassifierFocal(config)

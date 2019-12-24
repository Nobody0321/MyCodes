from networks.encoder import BiGru_Att
from networks.selector import Attention
from .Model import Model


class BiGru_ATT(Model):
	def __init__(self, config):
		super(BiGru_ATT, self).__init__(config)
		self.encoder = BiGru_Att(config)
		self.selector = Attention(config, config.encoder_output_dim)
		# self.classifier = ClassifierFocal(config)

from networks.selector import SelfSelectiveAttention
from networks.encoder import SelfMaxAttEncoder
from networks.classifier import Classifier2
from .Model import Model


class self_soft(Model):
    """
    sentence encoder: selfAttention, bag encoder soft attention, superbag encoder none
    """
    def __init__(self, config):
        super(self_soft, self).__init__(config)
        self.encoder = SelfMaxAttEncoder(config, input_dim=config.input_dim, output_dim=config.output_dim)
        self.selector = SelfSelectiveAttention(config, config.output_dim)
        self.classifier = Classifier2(config)

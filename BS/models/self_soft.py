from networks.selector import *
from networks.encoder import *
from .Model import Model


class self_soft(Model):
    """
    sentence encoder: selfAttention, bag encoder soft attention, superbag encoder none
    """
    def __init__(self, config):
        super(self_soft, self).__init__(config)
        self.encoder = SelfAttEncoder(config, input_dim=config.input_dim, output_dim=config.output_dim)
        self.selector = SoftAttention(config, config.output_dim)

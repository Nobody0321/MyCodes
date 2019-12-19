from networks.encoder import *
from networks.selector import *
from .Model import Model


class self_self(Model):
    """
    sentence encoder: selfAttention, bag encoder soft attention, superbag encoder none
    """
    def __init__(self, config):
        super(self_self, self).__init__(config)
        self.encoder = SelfAttEncoder(config, input_dim=config.input_dim, output_dim=config.output_dim)
        self.selector = SelfAttSelector(config, input_dim=config.output_dim, output_dim=60)

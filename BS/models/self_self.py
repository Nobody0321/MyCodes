from networks.encoder import SelfMaxAttEncoder
from networks.selector import SelfAttMaxSelector
from .Model import Model


class self_self(Model):
    """
    sentence encoder: selfAttention, bag encoder soft attention, superbag encoder none
    """
    def __init__(self, config):
        super(self_self, self).__init__(config)
        self.encoder = SelfMaxAttEncoder(config, input_dim=config.input_dim, output_dim=config.hidden_dim)
        self.selector = SelfAttMaxSelector(config, input_dim=config.hidden_dim)

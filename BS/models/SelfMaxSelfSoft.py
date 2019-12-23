from networks.encoder import SelfMaxAttEncoder
from networks.selector import SelfSoftAttSelector
from .Model import Model


class SelfMaxSelfSoft(Model):
    """
    sentence encoder: selfAttention, bag encoder soft attention, superbag encoder none
    """
    def __init__(self, config):
        super(SelfMaxSelfSoft, self).__init__(config)
        self.encoder = SelfMaxAttEncoder(config, input_dim=config.input_dim, output_dim=config.hidden_dim)
        self.selector = SelfSoftAttSelector(config, input_dim=config.hidden_dim)

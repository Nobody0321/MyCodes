from networks.encoder import SelfSoftAttEncoder
from networks.selector import SelfSoftAttSelector
from networks.classifier import Classifier2
from .Model import Model


class SelfSoftFocal(Model):
    """
    sentence encoder: selfAttention, bag encoder soft attention, superbag encoder none
    """

    def __init__(self, config):
        super(SelfSoftFocal, self).__init__(config)
        self.encoder = SelfSoftAttEncoder(config, input_dim=config.input_dim, output_dim=config.hidden_dim)
        self.selector = SelfSoftAttSelector(config, input_dim=config.hidden_dim)
        self.classifier = Classifier2(config)

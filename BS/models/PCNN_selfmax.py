from networks.encoder import *
from networks.selector import *
from .Model import Model


class PCNN_SelfMax(Model):
    def __init__(self, config):
        super(PCNN_SelfMax, self).__init__(config)
        self.encoder = PCNN(config)
        self.selector = SelfAttMaxSelector(config, config.hidden_size * 3)

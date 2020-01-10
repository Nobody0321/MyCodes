from networks.selector import *
from networks.encoder import *
from .Model import Model


class LayerAttention(Model):
    def __init__(self, config):
        super(LayerAttention, self).__init__(config)
        self.encoder = SelfAttEncoder(config, input_dim=config.input_dim, output_dim=config.hidden_dim)
        self.selector = SelfSelectiveAttention(config, config.hidden_dim)

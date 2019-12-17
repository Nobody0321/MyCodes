from networks.selector import *
from networks.encoder import *
from .Model import Model


class LayerAttention(Model):
    def __init__(self, config):
        super(LayerAttention, self).__init__(config)
        self.encoder = TransformerEncoder(config)
        self.selector = LayerAtt(config, config.d_feature)

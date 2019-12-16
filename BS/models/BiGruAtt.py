from networks.encoder import *
from networks.selector import *
from .Model import Model


class BiGruAtt(Model):
    def __init__(self, config):
        super(BiGruAtt, self).__init__(config)
        self.encoder = BiGru(config)
        self.selector = BagAttention(config, config.d_model)

from networks.encoder import *
from networks.selector import *
from .Model import Model


class BLSTM_ATT(Model):
    def __init__(self, config):
        super(BLSTM_ATT, self).__init__(config)
        self.encoder = BLSTM(config)
        self.selector = Bag_Attention(config)

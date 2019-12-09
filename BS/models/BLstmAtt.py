from networks.encoder import *
from networks.selector import *
from .Model import Model


class BLstmAtt(Model):
    def __init__(self, config):
        super(BLstmAtt, self).__init__(config)
        self.encoder = BLSTM(config)
        self.selector = Bag_Attention(config)
  
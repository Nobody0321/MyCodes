from networks.encoder import *
from .Model import Model


class DualAtt(Model):
    def __init__(self, config):
        super(DualAtt, self).__init__(config)
        self.encoder = TransformerEncoder(config, input_dim=config.input_dim, output_dim=config.output_dim)
        self.selector = TransformerEncoder(config, input_dim=config.output_dim, output_dim=config.output_dim)

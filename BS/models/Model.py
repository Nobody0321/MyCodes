from networks.encoder import *
from networks.classifier import *


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.encoder = None
        self.selector = None
        self.classifier = Classifier(config)

    def forward(self):
        embedding = self.embedding()
        sen_vec = self.encoder(embedding)
        logits = self.selector(sen_vec)
        return self.classifier(logits)

    def test(self):
        embedding = self.embedding()
        sen_embedding = self.encoder(embedding)
        return self.selector.test(sen_embedding)

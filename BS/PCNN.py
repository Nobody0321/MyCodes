import torch
import torch.nn as nn
import torch.nn.functional as functional


class PCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embs = nn.Embedding(config.vocab_size, config.word_dim)
        self.pos1_embs = nn.Embedding(config.pos_range, config.pos_dim)
        self.pos2_embs = nn.Embedding(config.pos_range, config.pos_dim)
        
        feature_dim = config.word_dim + config.pos_dim * 2
        rel_dim = feature_dim * 3

        self.masks = torch.LongTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.masks_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.data.copy_(self.masks)
        self.mask_embedding.weight.requires_grad = False

        self.relation_vec = nn.Parameter(torch.randn(config.relation_num, rel_dim))
        self.relation_bias = nn.Parameter(torch.randn(config.relation_num))
        
        all_filter_num = self.config.filters_num * len(self.config.filters)

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=config.filter_num, kernel_size=(k, word_feature_dim),
                            padding=(k // 2, 0)) for k in config.filters吗])
        self.dropout = config.dropout
        
        self.init_model_weight()
        self.init_word_embedding()

    def init_model_weight(self):
        nn.init.xavier_uniform_(self.relation_vec)
        nn.init.uniform_(self.relation_bias)
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.uniform_(conv.bias)

    def init_word_embedding(self):
        def p_2norm(path):
            v = torch.from_numpy(np.load(path))
            if self.config.norm_emb:
                v = torch.div(v, v.norm(2, 1).unsqueeze(1))
                v[v != v] = 0.0
            return v

        w2v = p_2norm(self.config.w2v_path)
        p1_2v = p_2norm(self.config.p1_2v_path)
        p2_2v = p_2norm(self.config.p2_2v_path)

        if self.config.use_gpu:
            self.word_embs.weight.data.copy_(w2v.cuda())
            self.pos1_embs.weight.data.copy_(p1_2v.cuda())
            self.pos2_embs.weight.data.copy_(p2_2v.cuda())
        else:
            self.pos1_embs.weight.data.copy_(p1_2v)
            self.pos2_embs.weight.data.copy_(p2_2v)
            self.word_embs.weight.data.copy_(w2v)

    def init_int_constant(self, num):
        if self.config.use_gpu:
            return torch.LongTensor([num]).cuda()
        else:
            return torch.LongTensor([num])
        
    def mask_piece_pooling(self, x, mask):
        x = x.unsqueeze(-1).permute(0,2,1,3)
        masks = self.masks_embedding(mask).unsqueeze(-2) * 100
        x = masks + x
        x = torch.max(x, dim=1)[0] - 100
        return x.view(-1, x.size(1) * x.size(2))

    def piecewise_max_pooling(self):
        
class Config(object):
    def __init__(self):

        self.data_path = './mini_dataset-1000'
        self.use_bag = True
        self.use_gpu = False
        self.is_training = True
        self.max_length = 120
        self.pos_num = 2 * self.max_length
        self.num_classes = 53
        self.hidden_size = 230
        self.pos_size = 5
        self.max_epoch = 15
        self.opt_method = 'SGD'
        self.learning_rate = 0.5
        self.weight_decay = 1e-5
        self.dropout = 0.1
        self.checkpoint_dir = './checkpoint'
        self.test_result_dir = './test_result'
        self.save_epoch = 1
        self.test_epoch = 1
        self.pretrain_model = None
        self.batch_size = 15
        self.sentence_len = 120
        self.word_size = 50
        self.window_size = 3
        self.epoch_range = None
        self.in_channels = self.word_size + self.pos_size * 2
        self.d_feature = 512
        self.d_ff = 2048
        self.d_model = 512
        self.n_heads = 8
        self.n_blocks = 6
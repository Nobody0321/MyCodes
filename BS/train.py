from config.Config import Config
import models
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
con = Config()
con.attn_dropout = 0.5
# con.set_pretrain_model("checkpoint/self_self-6")
con.use_gpu = True
con.data_dir = "./data"
# con.data_dir = "./mini_dataset-1000"
con.set_max_epoch(20)
con.batch_size = 8
con.output_dim = 230
con.d_ff = 512  # feed forward dim
con.n_heads = 5  # num of scale product attention heads
con.n_blocks = 1  # encoder blocks
con.set_learning_rate(0.0001)
con.set_weight_decay(0.0001)
# con.train_start_epoch = 7
con.save_iter = 4000
con.opt_method = "Adam"
con.load_train_data()
con.load_test_data()
con.set_train_model(models.self_self)
con.train()
